# The human launched the 2× H100 box, 2026-10-03 ~19:50Z

## Verbatim, in the lead session's chat

> 68.209.74.244 ;  2xh100 are back optimize training pipeline and start training.

Earlier, at ~19:03Z: "2xh100 are out", which meant Lambda's 2× H100 was sold out. Fable's ruling
was to wait. The 1× H100 fallback it named is superseded by this launch.

Then, through the AskUserQuestion tool, ~19:57Z, answering whether to install Python 3.12 on the
box: **"Yes, install it (Recommended)"**. The question named python3.12, python3.12-venv and
python3.12-dev from the deadsnakes package source Lambda's image already configures (8 packages,
~16 MB). The image's package lists were stale, so the install 404ed until the deadsnakes list
alone was refreshed. It installed 3.12.15, not the 3.12.12 the question named.

## The box

Read-only, 19:52:13Z:

| Item | Value |
|---|---|
| Host | `ubuntu@68.209.74.244`, Ubuntu 22.04.5 LTS, x86_64 |
| Booted | `uptime -s` 19:46:14Z. The box watch takes billing from 19:45:00Z, a minute early. |
| GPUs | 2× NVIDIA H100 80GB HBM3, 81,559 MiB each, driver 580.105.08 (the GH200's driver) |
| CUDA | nvcc 12.8 |
| Disk, RAM, CPUs | `/` has 5,434 GiB free (the floor is 600); 442 GiB RAM; 52 CPUs |
| Python | 3.10.12 only. 3.12.15 after the install above. |

`h100_setup.sh --preflight` passed at 19:54:09Z. The venv install started at ~19:54Z with the
nohup pattern, using the downloads the human approved at ~18:42Z
(`human-answers-2026-10-03-v5-launch.md`: the 70 wheels and causal_conv1d).

## What it answers, and how it is applied

| The words | Applied as |
|---|---|
| "2xh100 are back … start training" | The human's launch of the box chosen at ~16:26Z. It is not a new hardware choice: V5_BOX=h100x2, and V5_BOX_YES's 2× words stand. |
| "start training" | The queue's launch after the data is attested and the lane commit is deployed (deploy-plan.md phase B, step 10). These words go into `$Q/V5_LAUNCH_YES`. The spend ceiling is still V5_HUMAN_YES's ~$400 (launch.approved). |
| "optimize training pipeline" | The pre-registration freezes the training code: its recipe keys and code fingerprints. So no trainer or dataloader change goes into v5 (the lead's reply in chat). This round's optimization is in scheduling and the build. Seed 0's per-phase wall time and GPU utilization are read to choose v6's Rust port target. |

**Not covered.** Any GPU job before the queue's launch: a smoke or profiling run needs its own
yes. Terminating the box stays the human's action.
