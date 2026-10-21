"""Install, restart and health-check the `qd-metal-serve` launchd agent; say whether one runs.

    python3 -I tools/qd_serve_launchd.py install --binary B --release-dir D --release-sha H \\
        --socket S --stdout-log O --stderr-log E --launch-agents-dir A --launchctl L
    python3 -I tools/qd_serve_launchd.py restart ...same arguments...
    python3 -I tools/qd_serve_launchd.py health  ...same, minus --launch-agents-dir/--launchctl
    python3 -I tools/qd_serve_launchd.py render  ...same arguments as install; prints the plist
    python3 -I tools/qd_serve_launchd.py held --socket S

What this talks to. The daemon is `qd-metal-serve --release <dir> --socket <path>`
(`crates/qd-metal/src/bin/serve.rs`), which answers `qd serve`'s JSON-lines protocol on a Unix
socket (`crates/qd-runtime/src/main.rs:4`, `wire.rs` ControlOp). The health check uses only the two
control ops that protocol defines, `{"op":"ping"}` (liveness, touches no backend, cannot warm one)
and `{"op":"status"}` (`StatusReport`), plus an optional real decision request (`--warm-request`).
There is no pid file: the socket is the daemon's only liveness surface, so `held` reads it too.

Identity. "Release sha" is the release's `weight_hash` (`expected_identity.weight_hash` in
`release_manifest.json`), the figure `qd-metal-serve` prints at bind and a reply's
`identity.weight_hash` carries. It is not the sha of model.safetensors; `--verify-files` hashes
every file the manifest lists as well.

launchd. `--launch-agents-dir` and `--launchctl` are parameters on purpose: this script never
defaults to ~/Library/LaunchAgents or to /bin/launchctl, so running it is always a decision of the
caller's. The owner runs it for real; the tests use a temp directory and a fake launchctl.

Exit codes. 0 ok. 1 the health check failed (the reason is on stderr). 2 refused before touching
anything: bad arguments, a missing model, a sha that does not match the manifest. 3 launchctl
failed.
`held` exits 0 when no daemon is listening, 10 when one is (warm or cold: it warms on any request),
11 when that cannot be determined, so a caller must treat 11 as "held".
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import os
import plistlib
import shlex
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

EXIT_OK = 0
EXIT_HEALTH = 1
EXIT_REFUSED = 2
EXIT_LAUNCHCTL = 3
EXIT_HELD = 10
EXIT_UNKNOWN = 11

DEFAULT_LABEL = "com.bharath.lappi.qd-serve"
MAX_REPLY_BYTES = 1 << 20
LAUNCHCTL_TIMEOUT_S = 60.0
BIND_BANNER = "qd-metal-serve: listening on"
_HEX64 = frozenset("0123456789abcdef")


class Refused(Exception):
    """Exit 2: nothing was touched."""


class HealthFailed(Exception):
    """Exit 1."""


class LaunchctlFailed(Exception):
    """Exit 3."""


class Undetermined(Exception):
    """`held` could not tell."""


@dataclass(frozen=True)
class Spec:
    """Everything that goes into the plist, explicit: nothing is read from the environment."""

    label: str
    binary: Path
    release_dir: Path
    release_sha: str
    socket: Path
    stdout_log: Path
    stderr_log: Path
    idle_timeout_ms: int | None


def _is_sha(value: str) -> bool:
    return len(value) == 64 and set(value) <= _HEX64


def _absolute(path: Path, what: str) -> Path:
    if not path.is_absolute():
        raise Refused(f"{what} must be an absolute path, got {path}")
    return path


def load_manifest(release_dir: Path) -> dict[str, object]:
    manifest = release_dir / "release_manifest.json"
    if not manifest.is_file():
        raise Refused(f"{release_dir} is not a release: release_manifest.json is missing")
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise Refused(f"{manifest} cannot be read as JSON: {error}") from error
    if not isinstance(data, dict):
        raise Refused(f"{manifest} is not a JSON object")
    return data


def check_release(spec: Spec, verify_files: bool) -> None:
    """Refuse a release that is missing, or whose identity is not the one named."""
    if not _is_sha(spec.release_sha):
        raise Refused(
            f"--release-sha must be 64 lowercase hex characters, got {spec.release_sha!r}"
        )
    if not spec.binary.is_file() or not os.access(spec.binary, os.X_OK):
        raise Refused(f"the binary {spec.binary} is missing or not executable")
    if not spec.release_dir.is_dir():
        raise Refused(f"the release directory {spec.release_dir} does not exist")
    manifest = load_manifest(spec.release_dir)
    identity = manifest.get("expected_identity")
    recorded = identity.get("weight_hash") if isinstance(identity, dict) else None
    if recorded != spec.release_sha:
        raise Refused(
            f"release sha mismatch: --release-sha {spec.release_sha} but the manifest's "
            f"expected_identity.weight_hash is {recorded!r}"
        )
    files = manifest.get("files")
    if not isinstance(files, dict) or "model.safetensors" not in files:
        raise Refused("the manifest lists no model.safetensors")
    for name, entry in files.items():
        path = spec.release_dir / str(name)
        if not path.is_file():
            raise Refused(f"the release is missing {path}")
        if not isinstance(entry, dict):
            raise Refused(f"the manifest's entry for {name} is not an object")
        if path.stat().st_size != entry.get("bytes"):
            raise Refused(
                f"{path} is {path.stat().st_size} bytes; the manifest says {entry.get('bytes')}"
            )
        if verify_files and _sha256_file(path) != entry.get("sha256"):
            raise Refused(f"{path} does not hash to the manifest's sha256")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_plist(spec: Spec) -> bytes:
    arguments = [
        str(spec.binary),
        "--release",
        str(spec.release_dir),
        "--socket",
        str(spec.socket),
    ]
    if spec.idle_timeout_ms is not None:
        arguments += ["--idle-timeout-ms", str(spec.idle_timeout_ms)]
    document = {
        "Label": spec.label,
        "ProgramArguments": arguments,
        "RunAtLoad": True,
        "KeepAlive": True,
        "ThrottleInterval": 10,
        "StandardOutPath": str(spec.stdout_log),
        "StandardErrorPath": str(spec.stderr_log),
    }
    return plistlib.dumps(document, fmt=plistlib.FMT_XML, sort_keys=True)


def write_plist(spec: Spec, directory: Path) -> tuple[Path, bool]:
    """Write the plist atomically. Returns its path and whether the content changed."""
    if not directory.is_dir():
        raise Refused(f"the LaunchAgents directory {directory} does not exist")
    body = render_plist(spec)
    target = directory / f"{spec.label}.plist"
    changed = not target.is_file() or target.read_bytes() != body
    if changed:
        scratch = directory / f".{spec.label}.plist.{os.getpid()}"
        scratch.write_bytes(body)
        scratch.replace(target)
    for log in (spec.stdout_log, spec.stderr_log):
        log.parent.mkdir(parents=True, exist_ok=True)
    spec.socket.parent.mkdir(parents=True, exist_ok=True)
    return target, changed


class Launchctl:
    def __init__(self, command: list[str], domain: str, label: str) -> None:
        self.command = command
        self.domain = domain
        self.label = label

    @property
    def service(self) -> str:
        return f"{self.domain}/{self.label}"

    def run(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [*self.command, *args],
                capture_output=True,
                text=True,
                timeout=LAUNCHCTL_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise LaunchctlFailed(f"launchctl {' '.join(args)} could not run: {error}") from error

    def must(self, *args: str) -> None:
        done = self.run(*args)
        if done.returncode != 0:
            raise LaunchctlFailed(
                f"launchctl {' '.join(args)} exited {done.returncode}: "
                f"{(done.stderr or done.stdout).strip()}"
            )

    def loaded(self) -> bool:
        return self.run("print", self.service).returncode == 0


def ask(socket_path: Path, line: dict[str, object], timeout_s: float) -> dict[str, object]:
    """One request line, one reply line, over the daemon's socket."""
    payload = json.dumps(line, separators=(",", ":")).encode() + b"\n"
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(str(socket_path))
        conn.sendall(payload)
        reply = b""
        while not reply.endswith(b"\n"):
            chunk = conn.recv(65536)
            if not chunk:
                break
            reply += chunk
            if len(reply) > MAX_REPLY_BYTES:
                raise ValueError(f"the reply exceeded {MAX_REPLY_BYTES} bytes")
    value = json.loads(reply)
    if not isinstance(value, dict):
        raise ValueError("the reply is not a JSON object")
    return value


def _last_bind_banner(stderr_log: Path) -> str | None:
    try:
        text = stderr_log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    banners = [row for row in text.splitlines() if BIND_BANNER in row]
    return banners[-1] if banners else None


def health_once(spec: Spec, request_timeout_s: float, warm_request: Path | None) -> str:
    """One pass of the health check. Returns a summary, or raises HealthFailed with the reason."""
    try:
        pong = ask(spec.socket, {"op": "ping"}, request_timeout_s)
    except (OSError, ValueError) as error:
        raise HealthFailed(f"ping over {spec.socket} failed: {error}") from error
    if pong.get("status") != "ack" or pong.get("op") != "ping":
        raise HealthFailed(f"ping was answered with {pong!r}, not an ack")

    banner = _last_bind_banner(spec.stderr_log)
    if banner is None:
        raise HealthFailed(
            f"{spec.stderr_log} holds no '{BIND_BANNER}' line, so the daemon's own statement of "
            "the release it bound cannot be read"
        )
    if f"weight_hash {spec.release_sha}" not in banner or str(spec.release_dir) not in banner:
        raise HealthFailed(f"the daemon bound another release than expected: {banner.strip()}")

    if warm_request is not None:
        try:
            request = json.loads(warm_request.read_text(encoding="utf-8"))
            if not isinstance(request, dict):
                raise ValueError("the request is not a JSON object")
            answer = ask(spec.socket, request, max(request_timeout_s, 60.0))
        except (OSError, ValueError) as error:
            raise HealthFailed(f"the warm request failed: {error}") from error
        if answer.get("status") not in ("ok", "refused"):
            raise HealthFailed(f"the warm request was answered with a backend error: {answer!r}")

    try:
        status = ask(spec.socket, {"op": "status"}, request_timeout_s)
    except (OSError, ValueError) as error:
        raise HealthFailed(f"status over {spec.socket} failed: {error}") from error
    if status.get("status") != "status":
        raise HealthFailed(f"status was answered with {status!r}")
    if status.get("shutdown_requested") is True:
        raise HealthFailed("the daemon reports shutdown_requested")
    if status.get("warm") is True:
        backend = status.get("backend")
        if not isinstance(backend, dict) or backend.get("weight_hash") != spec.release_sha:
            raise HealthFailed(f"the warm backend's identity is not the release: {backend!r}")
        if backend.get("is_model") is not True or status.get("degraded") is True:
            raise HealthFailed("the warm backend is not a model, or answers degraded")
        state = "warm, identity verified on the wire"
    elif warm_request is not None:
        raise HealthFailed("the warm request was sent but the daemon reports it is not warm")
    else:
        state = "cold, identity verified from the bind banner only (send --warm-request to load it)"
    return f"healthy: {state}; runtime {status.get('runtime_version')}"


def health(spec: Spec, wait_s: float, request_timeout_s: float, warm_request: Path | None) -> str:
    """Retry the health pass until it holds or `wait_s` runs out (a restart needs a moment)."""
    deadline = time.monotonic() + wait_s
    while True:
        try:
            return health_once(spec, request_timeout_s, warm_request)
        except HealthFailed as failure:
            if time.monotonic() >= deadline:
                raise HealthFailed(f"{failure} (gave up after {wait_s:g}s)") from failure
            time.sleep(0.2)


def held(socket_path: Path, timeout_s: float) -> str:
    """Return a description when a daemon listens on the socket; raise Undetermined when unsure.

    No socket file, or one nothing listens on (a crash leaves the file behind), is "not held": no
    process owns a model. Anything else that is not a clean answer is Undetermined.
    """
    try:
        mode = socket_path.lstat().st_mode
    except FileNotFoundError:
        return ""
    except OSError as error:
        raise Undetermined(f"{socket_path} cannot be inspected: {error}") from error
    if not stat.S_ISSOCK(mode):
        raise Undetermined(f"{socket_path} exists and is not a socket")
    try:
        status = ask(socket_path, {"op": "status"}, timeout_s)
    except ConnectionRefusedError:
        return ""
    except OSError as error:
        if error.errno in (errno.ENOENT, errno.ECONNREFUSED):
            return ""
        raise Undetermined(f"{socket_path} did not answer a status request: {error}") from error
    except ValueError as error:
        raise Undetermined(
            f"{socket_path} answered something that is not a status: {error}"
        ) from error
    if status.get("status") != "status":
        raise Undetermined(f"{socket_path} answered {status!r}, not a status report")
    state = (
        "a model resident"
        if status.get("warm") is True
        else "no model loaded yet (it loads on any request)"
    )
    return f"a qd daemon is listening on {socket_path} with {state}"


def _spec(args: argparse.Namespace) -> Spec:
    return Spec(
        label=args.label,
        binary=_absolute(Path(args.binary), "--binary"),
        release_dir=_absolute(Path(args.release_dir), "--release-dir"),
        release_sha=args.release_sha,
        socket=_absolute(Path(args.socket), "--socket"),
        stdout_log=_absolute(Path(args.stdout_log), "--stdout-log"),
        stderr_log=_absolute(Path(args.stderr_log), "--stderr-log"),
        idle_timeout_ms=args.idle_timeout_ms,
    )


def _agent(args: argparse.Namespace, label: str) -> Launchctl:
    command = shlex.split(args.launchctl)
    if not command:
        raise Refused("--launchctl is empty")
    return Launchctl(command, args.domain, label)


def command_install(args: argparse.Namespace, restart_only: bool) -> str:
    spec = _spec(args)
    check_release(spec, args.verify_files)
    directory = _absolute(Path(args.launch_agents_dir), "--launch-agents-dir")
    control = _agent(args, spec.label)
    if restart_only:
        plist = directory / f"{spec.label}.plist"
        if not plist.is_file() or plist.read_bytes() != render_plist(spec):
            raise Refused(
                f"{plist} is absent or differs from these arguments; run install, not restart"
            )
        changed = False
    else:
        plist, changed = write_plist(spec, directory)
    loaded = control.loaded()
    if loaded and changed:
        control.must("bootout", control.service)
        control.must("bootstrap", control.domain, str(plist))
    elif loaded:
        control.must("kickstart", "-k", control.service)
    elif restart_only:
        raise LaunchctlFailed(f"{control.service} is not loaded; run install")
    else:
        control.must("bootstrap", control.domain, str(plist))
    return health(spec, args.health_wait_s, args.request_timeout_s, _optional(args.warm_request))


def _optional(value: str | None) -> Path | None:
    return Path(value) if value else None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, with_launchd: bool) -> None:
        p.add_argument("--binary", required=True, help="absolute path of qd-metal-serve")
        p.add_argument("--release-dir", required=True, help="the qd-export release directory")
        p.add_argument("--release-sha", required=True, help="the release's weight_hash (64 hex)")
        p.add_argument("--socket", required=True, help="absolute socket path")
        p.add_argument("--stdout-log", required=True)
        p.add_argument("--stderr-log", required=True, help="where the bind banner is read from")
        p.add_argument("--label", default=DEFAULT_LABEL)
        p.add_argument("--idle-timeout-ms", type=int, default=None)
        p.add_argument("--verify-files", action="store_true", help="sha256 every release file")
        p.add_argument(
            "--warm-request", default=None, help="a request JSON to send, loading the model"
        )
        p.add_argument("--health-wait-s", type=float, default=30.0)
        p.add_argument("--request-timeout-s", type=float, default=5.0)
        if with_launchd:
            p.add_argument("--launch-agents-dir", required=True)
            p.add_argument(
                "--launchctl", required=True, help="the launchctl command (may carry flags)"
            )
            p.add_argument("--domain", default=f"gui/{os.getuid()}")

    common(
        sub.add_parser("install", help="write the plist, bootstrap or kickstart, health-check"),
        True,
    )
    common(sub.add_parser("restart", help="kickstart -k the installed agent, health-check"), True)
    common(sub.add_parser("render", help="print the plist, change nothing"), False)
    common(sub.add_parser("health", help="health-check a running daemon"), False)
    probe = sub.add_parser("held", help="exit 0 free, 10 held, 11 unknown")
    probe.add_argument("--socket", required=True)
    probe.add_argument("--timeout-s", type=float, default=3.0)
    return parser


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "held":
            description = held(Path(args.socket), args.timeout_s)
            if description:
                print(description)
                return EXIT_HELD
            return EXIT_OK
        if args.command == "render":
            spec = _spec(args)
            check_release(spec, args.verify_files)
            sys.stdout.write(render_plist(spec).decode())
            return EXIT_OK
        if args.command == "health":
            spec = _spec(args)
            check_release(spec, args.verify_files)
            print(
                health(
                    spec, args.health_wait_s, args.request_timeout_s, _optional(args.warm_request)
                )
            )
            return EXIT_OK
        print(command_install(args, restart_only=args.command == "restart"))
        return EXIT_OK
    except Refused as error:
        print(f"qd_serve_launchd: refused: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except HealthFailed as error:
        print(f"qd_serve_launchd: health check failed: {error}", file=sys.stderr)
        return EXIT_HEALTH
    except LaunchctlFailed as error:
        print(f"qd_serve_launchd: launchctl failed: {error}", file=sys.stderr)
        return EXIT_LAUNCHCTL
    except Undetermined as error:
        print(
            f"qd_serve_launchd: cannot tell whether a daemon is running: {error}", file=sys.stderr
        )
        return EXIT_UNKNOWN


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
