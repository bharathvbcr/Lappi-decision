# W3b serving operations, 2026-10-21 (card ft-5dfe48e7, criteria 1-4)

Branch `lane/ft-5dfe48e7` from main `1bfdee2`. Not merged, not pushed. Nothing was installed on the
owner's Mac: no launchd agent was loaded, started or stopped, and `~/Library/LaunchAgents` was not
touched.

## Not in this lane

Criterion 5 (drift monitor: Haiku audit reader, per-slot abstention, permutation consistency and OOD
score, quarterly 30-posting owner-label refresh) and criterion 6 (one daemon with priority classes,
Jarvis at most 300 ms ahead of Vani batch at most 10 s, a stated budget under 32 GiB) are **not
done**, by the lane's scope. The card stays open for them.

## What changed

| Criterion | Where |
| --- | --- |
| 1. install, restart, health check | `tools/qd_serve_launchd.py` (Python stdlib, like `tools/mac_footprint.py`; it is an operator tool, not a hot path) |
| 2. guard and windows | `tools/mac_heavy.sh` ("Serving guard"), `docs/mac-serving-windows-2026-10-21.md` |
| 3. schedule | `docs/mac-schedule-2026-10-21.md` |
| 4. cloud cost | `docs/cloud-cost-estimate-2026-10-21.md` (DRAFT, NEEDS HUMAN YES) |
| tests | `python/tests/test_qd_serve_launchd.py` |

## The daemon the script installs

The launchd agent is `qd-metal-serve` (`crates/qd-metal/src/bin/serve.rs`), the model-bearing build of
`qd serve`. Plain `qd serve` answers `unavailable` to every request (no backend ships in it), so
installing it would be a daemon with no model. No pid file exists; the Unix socket is the liveness
surface. Health uses only protocol ops that exist (`wire.rs` ControlOp): `ping`, `status`, plus an
optional real request.

## Model path and release sha for the current release

There is **no v2 release**. The only release is the v0.1 preview
(`HANDOFF/lappi-v0.1-preview-release-notes-2026-10-06.md`, "a preview, not promoted"):

- release directory: `/Users/bharath/qd-campaign/preview-2026-10-05/release-lappi-v0.1-preview/`
  (model file `model.safetensors`, 3,763,692,048 bytes; sha256 `5ade5349b6d3fd28...e63848c`
  in its manifest)
- release sha, the `weight_hash`:
  `6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb` (read from the directory's own
  `release_manifest.json`, `expected_identity.weight_hash`: verified, and equal to the release notes)
- calibration hash `7e56b34e378907adccc7cfc0a086507d81c762ed6627694a0a59c3c977aeb677`
- binary: `target/release/qd-metal-serve` (README). **It is not built**: only a stale
  `target/debug/qd-metal-serve` exists in the main checkout, so the owner builds it first
  (`cargo build --release -p qd-metal`, a heavy job under `mac_heavy.sh`).

This preview serves general choice families only and refuses `code.defect_class` (release notes); it
is the only identity that exists, not a recommendation to serve it to Vani.

First command for the owner (a dry render; nothing is installed):

    python3 -I tools/qd_serve_launchd.py render --binary /Users/bharath/Code/research/Lappi-decision/target/release/qd-metal-serve --release-dir /Users/bharath/qd-campaign/preview-2026-10-05/release-lappi-v0.1-preview --release-sha 6f7b9ba73bca3a0b2e4ec9b2c0fc2aa6e7459f7e3286b26386633a222667dfcb --socket /Users/bharath/Library/Caches/qd/qd.sock --stdout-log /Users/bharath/Library/Logs/qd-serve/out.log --stderr-log /Users/bharath/Library/Logs/qd-serve/err.log

The real install adds `install --launch-agents-dir /Users/bharath/Library/LaunchAgents --launchctl
/bin/launchctl` (both explicit and required) and is the owner's decision. `render` was not run
against the real release in this lane.

## Unverified

- The script against a real `launchctl` and a real `qd-metal-serve`: not run (by instruction, and no
  release binary exists). Tests use a fake launchctl and a stub daemon that implements the control
  ops as `service.rs` encodes them, so a real reply shape that differs from `StatusReport` would
  not be caught.
- Whether `status.backend.weight_hash` is what the daemon puts on the wire for a warm model: read from
  `StatusReport`/`BackendIdentity`, not observed.
- Vani and Jarvis behaviour when the socket is absent during a training window.
- Unmarked training jobs are unguarded; wrappers that launch training do not set
  `MAC_HEAVY_TRAINING=1` yet.

## Gaps

See `gaps.jsonl` records `GAP-W3B-*` appended by this lane.
