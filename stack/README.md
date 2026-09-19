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

## 3. The lockfile resolves — and vLLM nearly broke it

`requirements.lock` is resolved for `x86_64-manylinux_2_28`, Python 3.12, and committed.
**206 packages.** The load-bearing pins:

```
torch==2.11.0+cu128    transformers==5.17.0    vllm==0.26.0
flash-linear-attention==0.5.2    causal-conv1d==1.7.0    triton==3.6.0
```

`transformers` is pinned to a **release**, not a main-branch SHA. Qwen3.5 has shipped in tagged
releases since v5.2.0 (2026-02-16), so the plan's "main moves daily" risk row is deleted rather than
mitigated.

**`--index-strategy unsafe-best-match` is required**, and that is a deliberate recorded choice, not a
convenience flag. The pytorch cu128 index carries `vllm` only at 0.9.0.1, which pins `torch==2.7.0`;
under default index precedence uv concludes torch 2.11 + vllm is unsatisfiable. The strategy lets it
consider all versions across both indexes. The dependency-confusion caveat uv warns about is real; it
is accepted because both indexes are first-party and well-known, and because the result is **pinned in
`requirements.lock`** rather than re-resolved at build time.

Regenerate with:

```bash
uv pip compile stack/requirements.in --python-platform x86_64-manylinux_2_28 --python-version 3.12 --index-strategy unsafe-best-match --output-file stack/requirements.lock
```

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

## 5. What this host cannot establish

Stated plainly so no lane reports these as passing:

- **The fla fast path cannot be verified here.** `is_flash_linear_attention_available()` gates on
  cuda/xpu/mlu with no MPS, and `is_causal_conv1d_available()` is CUDA-only. The gate runs as the
  first step of Session 0 and again at the head of the block.
- **Do not install `fla` in the Mac environment.** The Qwen3.5 modeling file dispatches through a bare
  `importlib.import_module("fla")` rather than the CUDA-gated helper, so a successful import here
  sends execution into Triton, which has no Metal backend.
- **The VM has 1.89 GiB of RAM.** Whether that survives the `torch` install layer is being measured by
  the first real build; if it OOMs the fix is `podman machine stop && podman machine set --memory 8192
  && podman machine start`. That change is the host owner's to make, so it is flagged rather than
  applied.

## S1 status

| Item | State |
| --- | --- |
| podman substituted for docker, cross-arch verified | **done**, measured |
| Base image tag verified to exist (2 platform entries) | **done** |
| Lockfile resolved and committed | **done**, 206 pins |
| Containerfile authored | **done** |
| Fail-open grep replaced by a positive assertion | **done**, `not_run` path tested |
| amd64 image builds end to end | **in progress** on this host |
| Fast path confirmed live | **not run** — needs CUDA; Session 0 |
| aarch64 image | **not needed** — GH200 is the only aarch64 target and is out of capacity |
