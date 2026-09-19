# Local environment audit — the Mac the first three weeks run on

Measured 2026-09-19. Every row is an observed fact with the command that produced it, or it is marked
UNVERIFIED. Re-measure before relying on any of it.

## Toolchain

| Thing | Observed | Command |
| --- | --- | --- |
| rustc / cargo | 1.98.0 (2026-08-18) | `rustc --version` |
| Python | 3.14.7 (Homebrew) | `python3 --version` |
| uv | 0.11.28 | `uv --version` |
| git | 2.54.0 (Apple Git-157) | `git --version` |
| **docker** | **NOT INSTALLED** | `which docker` -> not found |
| podman | 6.1.2, machine `podman-machine-default` (applehv, 9 CPU, **2 GiB RAM**, 100 GiB disk) | `podman --version`, `podman machine list` |
| torch (`~/.venvs/ml`) | 2.12.1, **MPS available: True** | `python -c "import torch; torch.backends.mps.is_available()"` |

**S1 impact.** The plan says the container image is "built with buildx for x86". Docker is absent;
podman is the substitute (user-directed). Two things remain UNVERIFIED and gate calling S1 green:

1. Cross-arch build (aarch64 host -> `linux/amd64` image) inside the applehv machine. Podman needs
   qemu-user emulation registered in the VM; whether this machine has it is untested.
2. **2 GiB of VM RAM is very likely insufficient** to build a torch + CUDA image. `podman machine set
   --memory` raises it, but the machine must be stopped to do so. Recorded as `GAP-S1-NO-DOCKER`.

## Models already on disk

From `~/.cache/huggingface/hub`:

| Model | Bearing on the plan |
| --- | --- |
| `mlx-community/Qwen3.8-27B-4bit` | **The teacher the plan names, already local.** The rubric iteration and the kappa >= 0.6 check on 50 items need no download and no rental |
| `mlx-community/Qwen3.8-27B-MTP-4bit`, `-nvfp4`, `-MTP-nvfp4` | Variants; the MTP ones carry the multi-token-prediction head |
| `incoai/Qwen3.8-27B-DFlash2` | — |
| `mlx-community/gemma-4-31B-it-assistant-bf16`, `z-lab/gemma-4-31B-it-DFlash` | The plan's alternative teacher, also local |
| `google/gemma-4-E4B-it-assistant`, `mlx-community/gemma-4-e4b-it-4bit` | — |

**Qwen3.8-27B existing in four local builds is direct evidence for the plan's claim that Qwen3.8 is
real.** Whether it "shipped nothing dense under 27B" is a separate claim, still being verified.

**Not present: any Qwen3.5-2B checkpoint.** The backbone must be downloaded before week 1's
"MPS forward of Qwen3.5-2B verified and timed" can run. At ~4 GB in bf16 this is minutes, not a
blocker — but it is a prerequisite nobody has done yet, and the plan's day-one item silently assumes
the weights are there.

## Disk

227 GiB free of 926 GiB (75% used).

The plan's pool pull wants "~50-100 GB" for the Rust/Go/Swift/Python/TS slices of AgentPack (371 GB
total). That fits, but not comfortably once checkpoints (2B bf16 ~4 GB each, times CPT + prune + 3 FT
seeds + ablations) and pre-tokenized memmap shards land beside it. **Budget the disk before the pull,
not during it**: a pull that dies at 90% full leaves a partial shard set that looks complete.

## What this environment cannot do

Stated plainly so no lane reports these as passing:

- **No CUDA.** Every GPU suite in the plan that needs CUDA is `not_run` here, with that reason, never
  green. This covers S3 (throughput/MFU), S5 (bit-exact resume), S8, S10, and all of session 1.
- **flash-linear-attention / causal-conv1d**: Triton-based and CUDA-source-built respectively; neither
  is expected to work on MPS. The plan's own day-one item — whether `transformers` at the pinned SHA
  runs Qwen3.5-2B forward **on MPS without fla and causal-conv1d** — is explicitly flagged unconfirmed
  in the plan and remains so here. It decides whether the depth probe is a 2-hour torch item or an
  MLX item.
- **No Metal kernel work is authorized yet.** K1-K7 start only after the shipping gate passes on
  Lambda. The parity suite, K7, K3 and K2 are the week-2/3 exception the plan carves out.
