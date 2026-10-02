# rung0 `04ab105a…24f7`: its source of record (L-cuda-M1, 2026-10-01)

The deployed rung-0 binary for the GH200 is pinned on the box by its file sha256
(`box_q_rung0.sh` refuses a mismatch). This file and the tarball beside it are the **source of
record** for that binary. The lead ruled on 2026-10-01 that rung0 is frozen in that sense: the
bytes on the box and a reproducible source record. It is not frozen in the sense that the crate
stops moving. Any future rung0 redeploy is a new sha, a new pin and a new record.

## The snapshot

- File: `AUDIT/ojas-training-2026-10-01/l-cuda-rung0-source-snapshot.tar.gz`, 114,904 bytes.
- **sha256 `d536b031c5bb3157099e419a3a2103dee39750def261a3ff4450fa5d84274069`**.
- Paths are relative to `/Users/bharath/Code/research/ojas`. It holds:
  - `ojas-qwen35-cuda/{Cargo.toml, Cargo.lock, README.md, src/**}` (22 source files);
  - `ojas-core/{Cargo.toml, src/**}`: rung0's crate depended on it then;
  - `Cargo.toml`, the ojas workspace root: `ojas-core` inherits `edition`, `license`,
    `rust-version` and `publish` from it.
- `tests/**` is not in it. Test targets are not part of the `rung0` unit.

**How it was verified [V].** Taken at 19:46 CDT. A fresh rebuild from the live tree at 19:48
CDT, into a new target dir, gave `04ab105a…24f7`. The tarball was extracted and `diff -r`
against the live tree: `src/`, `Cargo.toml`, `Cargo.lock` and `ojas-core/src` were identical.

## Rebuilding it

The **checkout path matters**: the same sources in a copy at another path built `9a040b73…`
(measurement C below). Extract the tarball at `/Users/bharath/Code/research/ojas`, then:

    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=/Users/bharath/qd-campaign/sysroot-aarch64-linux-gnu/link.sh
    export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUSTFLAGS="-C linker-flavor=gcc"
    cargo build --offline --release --target aarch64-unknown-linux-gnu --features cuda --bin rung0 \
      --manifest-path /Users/bharath/Code/research/ojas/ojas-qwen35-cuda/Cargo.toml \
      --target-dir <any new directory>
    shasum -a 256 <target-dir>/aarch64-unknown-linux-gnu/release/rung0

**The target dir does not matter.** Two fresh target dirs both gave `04ab105a…24f7`:
`…-m1-baseline` and `…-m1-snapshot-check`, under `/Users/bharath/qd-campaign/`.

**The toolchain the hash depends on:**
- `rustc 1.98.0 (88d9e12ae 2026-08-18)`, `cargo 1.98.0 (797e8a9bc 2026-08-05)`, at
  `~/.rustup/toolchains/stable-aarch64-apple-darwin`. Its bundled `ld.lld` is the linker.
- `link.sh`, sha256 `93e78e26b0df7be7e755517b497f2a6e4ad82bfa7b784112b3c90399043c73e8`.
  It runs `/usr/bin/clang --target=aarch64-unknown-linux-gnu`, with the box's glibc 2.39 files as
  the sysroot.
- cudarc 0.19.10 from crates.io, with checksum `7359bf1d…` in the snapshot's `Cargo.lock`.
- libloading 0.9.0, cfg-if 1.0.5, windows-link 0.2.1.

## Why any library change moves the rung0 sha (three measurements)

All three ran on 2026-10-01 between 19:42 and 19:47 CDT, with the toolchain above.

| # | Change | Where | rung0 sha256 | Against |
| --- | --- | --- | --- | --- |
| A | none: a fresh rebuild | real checkout path | `04ab105a…24f7` | the deployed `04ab105a…24f7`: **equal** |
| B | `ojas-core` removed from `[dependencies]` | real checkout path, restored byte-for-byte afterwards | `07d156dd196dd979…` | `04ab105a…`: **differs** |
| C0 | none | a copy of the tree at another path | `9a040b735b42119e…` | the copy's own baseline |
| C1 | one unreferenced 3-line `pub mod` added to `lib.rs` | the copy | `a25fc8df6f38b7a3…`, with 1,138,626 bytes differing | C0: **differs** (code layout moves) |
| C2 | the same module behind `#[cfg(feature = "m1")]`, built without `m1` | the copy | `d982032bf845d002…`, with 198 bytes differing, all inside `.strtab` | C0: **differs** |
| C3 | a new feature declared in `Cargo.toml` only | the copy | `9a040b73…` | C0: **equal** |

**What B and C mean.**
- B: a dependency is part of cargo's `-C metadata` hash for the library and the binary, so every
  mangled symbol moves.
- C1: an added module changes the library's codegen-unit partitioning.
- C2: a line that compiles to nothing still moves rung0's symbol-name hashes.

**Conclusion [V for these cases, I in general].** rung0 rebuilt from this crate stays
`04ab105a…` only while `lib.rs` and every compiled `src/` file stay byte-identical. M1,
L-cuda-gdn and L-cuda-small all add modules, so a rebuilt rung0 differs from the deployed one
after them. It differs **through the library's bytes only**: `src/bin/rung0.rs` is unchanged.

After `ojas-core` was removed (the lead's ruling), the snapshot is the only record of the
manifest that produced `04ab105a`.
