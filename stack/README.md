# S1 — the pinned environment

The plan's S1 is *"One lockfile (uv) and a container image ... A 30-second dry run greps its own log
for the fla fallback warning and exits non-zero if present. Done when: image builds on x86 and
aarch64; dry run green on both."*

Two things about that are wrong, and one is an environment fact. All three are handled here.

## 1. podman, not docker

Docker is not installed on this host. `podman` 6.1.2 is, with an `applehv` machine.

**Measured 2026-09-19:**

| Question | Answer | How it was established |
| --- | --- | --- |
| Is cross-arch emulation available? | **Yes** — `qemu-x86_64` is registered in the VM's `binfmt_misc` | `podman machine ssh "ls /proc/sys/fs/binfmt_misc/"` |
| Can it pull an amd64 image? | Yes | `podman pull --platform linux/amd64 alpine:3.20` |
| Can it *execute* amd64? | **Yes** — reports `x86_64`, vs `aarch64` natively | `podman run --platform linux/amd64 alpine uname -m` |
| How slow is emulation? | **~1.4x** on a compile+float benchmark (3.09 s vs 2.24 s, images cached) | `gcc -O2` plus a 2e8-iteration loop, both platforms |
| Can emulation run **`nvcc`**? | **NO.** `qemu: uncaught target signal 11 (Segmentation fault)` | a real `causal-conv1d` source build |
| VM resources | 9 CPUs, **1.89 GiB RAM**, 100 GiB disk | `podman info` |

The `buildx` in the plan becomes:

```bash
podman build --platform linux/amd64 -t qd-train:s1 -f stack/Containerfile stack/
```

`podman manifest` covers the multi-arch case if aarch64 is ever needed. It is **not** needed for the
plan as it stands: GH200 was the only aarch64 target, it showed out-of-capacity, and the plan's own
guidance is not to wait for it.

## 2. The dry-run grep is a check that cannot fail — replaced

`stack/verify_fast_path.py` replaces it. Reasoning in full at the top of that file and in
`docs/plan-corrections.md` SAFETY-1; the short version:

- the fla fallback warning fires on **first forward, not at import**, so a 30-second dry run that
  never runs a forward sees nothing either way;
- it is **suppressed entirely under `torch.compile`** by a `not is_torchdynamo_compiling()` guard,
  and S3 explicitly tries `torch.compile`;
- so on a compiled run the grep passes whether or not the fast path is live, while the job proceeds
  at **74% slower per step** (1.22 vs 0.70 s/step; the transformers source puts the
  `chunk_gated_delta_rule` gap at "more than an order of magnitude on an H100").

The replacement asserts a **presence**, not an absence: it discovers the fast and fallback symbols in
the installed modeling module, wraps both with counters, runs a real forward, and reports which one
executed. It refuses to report a result when it could not bind both paths, when neither counter moved,
or when the device is not CUDA — and `--require-fast` exits non-zero on `not_run` as well as on
failure, because an unchecked environment is not a clean one.

Verified on this Mac: the checker correctly returns `not_run` (no Qwen3.5 module importable) and
`--require-fast` exits **1**.

## 3. One lockfile does not hold. vLLM and the trainer need separate images.

The plan's S1 says *"One lockfile (uv) and a container image"* holding
`transformers`, `flash-linear-attention`, `causal-conv1d`, torch **and** `vLLM for the
teacher`. **That does not resolve into a working image**, and the way it fails is quiet.

The combined lockfile pinned `torch==2.11.0+cu128` on paper while simultaneously pulling
**both CUDA wheel families**:

```
nvidia-cuda-runtime==13.4.92        <- CUDA 13
nvidia-cuda-runtime-cu12==12.8.90   <- CUDA 12
```

At install time it produced `torch 2.14.0+cu130`, and `causal-conv1d`'s source build
stopped it:

```
RuntimeError: The detected CUDA version (12.8) mismatches the version
that was used to compile PyTorch (13.0)
```

The cause is that **vLLM drags in `torchaudio`, `torchvision` and `torchcodec`**, each
carrying its own torch constraint, and `--index-strategy unsafe-best-match` then lets
them mix across indexes — doing precisely what uv warns that flag can do.

**Measured, by removing vLLM and re-resolving:**

| | Combined | Training only |
| --- | --- | --- |
| Packages | 206 | **72** |
| torch | claims `2.11.0+cu128`, installs `2.14.0+cu130` | `2.11.0+cu128` |
| CUDA 13 wheels | 12 | **0** |
| CUDA 12 wheels | mixed | 15 |
| `unsafe-best-match` needed | yes | **no** |

And resolving the teacher on its own gives `vllm==0.29.0` with `torch==2.13.0` and the
**CUDA 13** family. The two environments genuinely want different CUDA majors, so they
want different base images. This is not a preference.

**So there are two lockfiles and two images:** `train.lock` (72 pkgs, CUDA 12, needs the
`-devel` base because causal-conv1d compiles) and `teacher.lock` (196 pkgs, CUDA 13,
prebuilt wheels only). Splitting them costs nothing the plan cares about: the teacher
runs once for ~40 minutes at the head of the block and the trainer for ~30 hours after
it, so they are never resident together — the plan's own session ordering already
separates them in time.

The upside is not only that it builds: dropping `unsafe-best-match` **removes** the
dependency-confusion caveat rather than accepting it.

## 3a. Historical note — the index-precedence trap

Kept because anyone re-deriving a combined lockfile hits this first and draws the wrong
conclusion from it. Under *default* index precedence, uv calls the combined set flatly
unsatisfiable:

```
Because only vllm==0.9.0.1 is available and vllm==0.9.0.1 depends on torch==2.7.0 ...
and because you require torch==2.11.0, we can conclude that your requirements
are unsatisfiable.
```

That message is misleading: the pytorch cu128 index simply *carries* only an ancient
vLLM, and uv will not look past the first index that has a package. The tempting fix is
`--index-strategy unsafe-best-match`, which makes the resolution succeed — and produces
the silently-broken two-CUDA-family lockfile described above.

**The correct fix is the split, not the flag.** Neither `train.in` nor `teacher.in`
needs `unsafe-best-match`.

## 3b. The pins, and why transformers is a release

`train.lock` — 72 packages, resolved for `x86_64-manylinux_2_28` / Python 3.12:

```
torch==2.11.0+cu128    transformers==5.17.0    triton==3.6.0
flash-linear-attention==0.5.2    causal-conv1d==1.7.0
```

`teacher.lock` — 196 packages, same platform:

```
vllm==0.29.0    torch==2.13.0    transformers==5.17.0    triton==3.7.1
```

`transformers` is pinned to a **release**, not a main-branch SHA. Qwen3.5 has shipped in
tagged releases since v5.2.0 (2026-02-16), so the plan's "main moves daily" risk row is
deleted rather than mitigated. `torch` is deliberately **unpinned** in `teacher.in`:
vLLM's own constraint decides, and pinning it to a value vLLM disagrees with is exactly
what broke the combined lockfile.

Regenerate either with:

```bash
uv pip compile stack/train.in --python-platform x86_64-manylinux_2_28 --python-version 3.12 --emit-index-url --output-file stack/train.lock
```

`--emit-index-url` is not optional. Without it uv strips the `--extra-index-url` from
the output, and the lockfile no longer describes its own resolution: the build then
cannot find `torch==2.11.0+cu128` at all. That was the second real build failure.

## 4. What the first real build found

The first build of `Containerfile` **failed**, and usefully:

```
error: The interpreter at /usr is externally managed  (PEP 668)
Error: building at STEP "RUN uv pip install --system ..." : exit status 2
```

Ubuntu 24.04 marks its system Python externally-managed, so `uv pip install --system`
refuses outright. Fixed by installing into a uv-managed venv at `/opt/venv` and putting it on
`PATH`. `--break-system-packages` would also have worked; a venv was chosen because mixing
apt-managed and pip-managed site-packages in an image that later builds CUDA extensions is how a
rebuild stops reproducing.

**A process note worth keeping.** The failing build was launched as `podman build ... | tail -40`,
and the pipeline reported **exit 0** — the exit status of `tail`, not of the build. A build that
failed looked like a build that passed. Subsequent builds redirect to a file and report
`BUILD_EXIT=$?` directly. This is the same class of error as the fla log-grep: a check whose success
signal is not actually attached to the thing being checked.

## 4a. qemu cannot run `nvcc` — and that is why torch is pinned to 2.10

The sixth build failure was not a resource problem:

```
qemu: uncaught target signal 11 (Segmentation fault) - core dumped
```

**The x86 emulation layer segfaults running `nvcc`.** Ordinary compilation emulates fine — the ~1.4x
figure above was measured on `gcc` — but `nvcc` compiling six GPU architectures does not survive it.
A general "cross-arch works here" conclusion drawn from the `gcc` benchmark is **wrong for the one
compiler this image needs**, and that inference was made here before it was tested.

The fix is not more emulation. `causal-conv1d` ships **prebuilt wheels** on its GitHub releases, one
per `(cuda x torch x cxx11abi)` combination, and its `setup.py` downloads the matching one instead of
compiling — when one exists. At torch 2.11 it looked for:

```
causal_conv1d-1.7.0+cu12torch2.11cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

which **is not in the v1.7.0 release**: the published cu12 wheels stop at `torch2.10`. So it fell
back to source, and qemu killed it.

**Pinning torch to 2.10 turns the most fragile step in the image into a wheel download.** That is a
better choice than the plan's 2.11 independent of this Mac: it takes a from-source CUDA compile off
the critical path of a $1,085 block, where a build failure costs $32/hour to debug and the `-devel`
base, the six `-gencode` targets and the OOM exposure all come with it.

The deviation from the plan's stated `torch 2.11 cu128` is deliberate and recorded here and in
`docs/plan-corrections.md` STACK-1.

## 4b. Disk — and why the number you read after a failure is the wrong number

The eighth build failed with `no space left on device`, twice in one log: writing
`/root/.cache/uv/archive-v0/.../nvidia/nccl/lib/libnccl.so.2` inside the container, and writing
`.../containers/storage/overlay-layers/tmp/...` on the VM. Three findings came out of it, and the
first nearly sent the diagnosis to the wrong machine.

**The host was never the constraint.** The Mac had 173 GiB free and the machine image was already
fully allocated at its configured size. The VM filled, not the host — worth checking in that order,
because a sparse VM image on a full host produces guest ENOSPC while the guest still reports free
space, and that failure looks identical from inside.

**Free space measured after a failed build understates peak demand**, by exactly the size of whatever
podman rolled back on the way out. The guest read 51G of 100G used *after* the failure — which looks
like ample headroom and is not, because the five failed-build layers had already been reclaimed by
then. Peak is the base (9.41 GB) plus the install layer plus a transient tar of that same layer while
buildah commits it: **the layer is paid for twice at the moment of commit.**

**`PIP_NO_CACHE_DIR=1` was inert.** Every install here goes through `uv pip`, and uv does not read
that variable — its switch is `-n/--no-cache`, env `UV_NO_CACHE`. The wheel cache was being committed
into the image. Fixed at the `ENV` declaration; recorded as `GAP-STACK-UV-CACHE-VAR-INERT`. It is the
third instance this session of one defect class: a guard that never engaged, reading identically to
one that ran and passed.

**Growing the machine takes two steps, not one.** `podman machine set --disk-size 160` grows the
block device and stops there:

```
vda    160G
└─vda4  99.5G  /var      <- unchanged, and df is unchanged with it
```

Fedora CoreOS grows the root partition on *first* boot, so a later resize leaves the partition behind
and `df` reports no change at all — the resize looks like it silently failed. The filesystem is xfs:

```bash
podman machine ssh "sudo growpart /dev/vda 4"
podman machine ssh "sudo xfs_growfs /var"
```

Result: 160G total, 119G free. Only the five dangling layers this session produced were removed
(10 GB reclaimed); the pre-existing images and the 14.96 GB of named volumes were left alone. A prune
that reclaims someone else's data to unblock your own build is not a fix.

## 5. What this host cannot establish

Stated plainly so no lane reports these as passing:

- **The fla fast path cannot be verified here.** `is_flash_linear_attention_available()` gates on
  cuda/xpu/mlu with no MPS, and `is_causal_conv1d_available()` is CUDA-only. The gate runs as the
  first step of Session 0 and again at the head of the block.
- **Do not install `fla` in the Mac environment.** The Qwen3.5 modeling file dispatches through a bare
  `importlib.import_module("fla")` rather than the CUDA-gated helper, so a successful import here
  sends execution into Triton, which has no Metal backend.
- **Builder memory — resolved, and the interesting half is not the RAM.** The 1.89 GiB VM was confirmed
  insufficient by a real build: `c++: fatal error: Killed signal terminated program cc1plus`, the OOM
  killer, during `causal-conv1d`'s CUDA extension compile. Note the message names the *compiler*, so
  it reads as a toolchain problem rather than a resource one.

  Fixed in two places. The builder VM was raised to **12 GiB** (`podman machine set --memory 12288`;
  revert with `2048` — nothing was running on it and the host has 64 GiB). More importantly
  **`MAX_JOBS=4` is set in the Containerfile**, because peak memory during a CUDA extension build is
  `MAX_JOBS`-shaped, not total-RAM-shaped: the build fans out one job per core, each translation unit
  can take a couple of GB, and raising RAM alone only moves the core count at which it dies. Putting
  the bound in the image means the constraint travels with it, so the same Containerfile builds on a
  small CI runner.

## 6. The teacher image

`stack/Containerfile` has pointed at "stack/README.md section 6" since it was written. There was
no section 6, and there was no teacher image either — only `teacher.lock`, resolved and committed
with nothing that consumed it. A dangling cross-reference is a cheap tell for a missing deliverable,
and this one was accurate.

```bash
podman build --platform linux/amd64 -t qd-teacher:s1 -f stack/Containerfile.teacher .
```

**What differs from the trainer, and why:**

| | Trainer | Teacher |
| --- | --- | --- |
| Base | `nvidia/cuda:12.8.1-**devel**-ubuntu24.04` | `nvidia/cuda:13.0.1-**runtime**-ubuntu24.04` |
| CUDA family | 12.8 (`nccl-cu12`, `cudnn-cu12`) | 13 (`nvidia-cuda-runtime==13.0.96`, `nccl-cu13`, `cudnn-cu13`) |
| Pins | 71 | 196 |
| Compiles anything? | yes — `causal-conv1d` | **no** — prebuilt wheels end to end |
| Gate | `verify_fast_path.py --require-fast` | `verify_teacher.py --require-ready` |

`-runtime` rather than `-devel` is the substantive choice: nothing in `teacher.lock` builds from
source, so a `-devel` base would carry ~6 GB of unused toolchain. Note that `nvidia-cuda-nvcc==13.4.92`
*is* in the lock — flashinfer JITs kernels at run time and takes nvcc from that wheel. A `-devel`
base would put a **second** nvcc at a different version on the path, which is exactly the sort of
quiet duplication that decides a run's behaviour six hours in.

The base tag was checked with `podman manifest inspect` before being written down: `13.0.1-runtime-ubuntu24.04`
publishes arm64 and amd64, and the floating `13.0-runtime-ubuntu24.04` tag **does not exist**
(`manifest unknown`), so the three-component version is required rather than stylistic.

### The context is the repo root, for both images

Both Containerfiles build from the **repo root**, not `stack/`, and `.containerignore` keeps that
context small (target/ alone is 1.5 GB against python/ at 1.7 MB).

This was forced by a defect worth recording. The ninth build tagged `qd-train:s1` clean — 14/14
steps, `x86_64` verified, `torch-2.10.0+cu128` verified, the `causal_conv1d_cuda` extension present.
Running it produced:

```
ModuleNotFoundError: No module named 'qd_train'
```

The gate imports `qd_train.tristate`, and a `stack/`-only context cannot reach `python/`. **The image
built, tagged, and verified as an image, while the one thing inside it that exists to be run could
not start.** Every property checked was a property of the artifact; none was a property of the
artifact doing its job.

The fix is layout, not a code branch: the image mirrors the repo — `stack/` beside `python/` under
`/opt/qd` — so the gate's `Path(__file__).resolve().parents[1] / "python"` is correct in a checkout
and in the container, with no container-only path for a future edit to break. Recorded as
`GAP-STACK-GATE-CANNOT-IMPORT`.

## S1 status

| Item | State |
| --- | --- |
| podman substituted for docker, cross-arch verified | **done**, measured |
| Base image tag verified to exist (2 platform entries) | **done** |
| Lockfile resolved and committed | **done** — `train.lock` 71 pins, `teacher.lock` 196. The single 206-pin resolution is what section 3 rejected |
| Containerfile authored | **done**, both — `stack/Containerfile` (trainer) and `stack/Containerfile.teacher` |
| Fail-open grep replaced by a positive assertion | **done** — `python/tests/test_stack_gates.py`, 15 tests. This row previously read "`not_run` path tested" while **no test referenced either gate script**; the claim was false when written. The fail-open property is now mutation-checked: making `--require-ready` accept `NotRun` fails 2 tests |
| Teacher image (`Containerfile.teacher`) builds | **done** — `localhost/qd-teacher:s1`, **10.7 GB** against the trainer's 17.4 GB, 14/14 steps, first attempt. Verified inside the image: `x86_64`, `vllm-0.29.0`, `torch-2.13.0`, `transformers-5.17.0`, all matching `teacher.lock`. Its gate runs: exits **0** reporting, **1** under `--require-ready` |
| Teacher readiness gate | **done** — `stack/verify_teacher.py`, same tri-state contract. Reaching its `cuda.is_available()` branch is itself evidence: torch imported, vLLM imported, and `vllm.__version__` matched `teacher.lock`, because a mismatch returns before that point |
| Teacher readiness confirmed | **not run** — needs CUDA; Session 0. `not_run`, not a pass |
| amd64 image builds end to end | **done** — `localhost/qd-train:s1`, 17.4 GB, 14/14 steps, on the ninth attempt. Verified inside the image: `uname -m` = `x86_64`, `torch-2.10.0+cu128`, and `causal_conv1d_cuda.cpython-312-x86_64-linux-gnu.so` present, so the prebuilt wheel carried its compiled extension rather than nvcc running under qemu |
| The trainer image's gate actually runs | **done** — the tenth build. Ninth tagged clean with a gate that died on `ModuleNotFoundError`; see section 6. Now reports `state: not_run`, `reason: device is 'cpu', not cuda`, and the **container** exits **1** under `--require-fast` |
| Fast path confirmed live | **not run** — needs CUDA; Session 0. This is a `not_run`, not a pass, and the gate exits non-zero for exactly that reason |
| aarch64 image | **needed, NOT BUILT** — a GH200 (96 GB HBM3, aarch64 Grace) was rented on 2026-09-20, so the "out of capacity" premise of this row is gone. Only `linux/amd64` has ever been built, and that image cannot run on it. The hard part is already answered: `tools/verify_lock_portability.py --platform aarch64-manylinux_2_28` reports **portable to aarch64** — all 71 `train.lock` and 196 `teacher.lock` pins resolve identically to the committed x86_64 lock, the `torch 2.10.0+cu128` cp312 aarch64 wheel exists, and **2** causal-conv1d cp312/torch2.10 aarch64 wheels exist, so the from-source fallback this file calls "the single most fragile step in the image" does **not** return. What remains is `podman build --platform linux/arm64` and verifying `uname -m` = `aarch64` inside it |
