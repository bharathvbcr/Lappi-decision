"""Counting what a run is billed for -- the half of the price that needs torch.

:class:`qd_train.run_control.CostEstimate` is **torch-free on purpose**.
``docs/training-contract.md``: *"Format, mutation, prompt assembly, splitting and ledger
bookkeeping are torch-free, and run in CI on any machine."* So it takes ``n_gpus`` as a
number and cannot go and count them, and the counting has to live out here among the tools
that already import torch.

## Why its own module rather than either tool

Two tools run on rented hardware and need this: ``real_ft_run.py`` and
``rung0_real_run.py``. A tool importing the other tool would pull a text-tower loader in to
answer a one-line question, and a second copy would be a clone of the kind ``repo_git.py``
was written to remove -- that module exists because ``devmap_clones`` reported its two
callers' private copies as an Exact pair.

## Why counting rather than a flag

``n_gpus`` is one of the two values that turn rule 4 off when they are wrong, and it is the
one the caller has no reason to know: the price list charges for the GPUs the box has, and
the box is the authority on that. ``--usd-per-hour`` must be stated because it is a fact
about a contract; ``n_gpus`` must be counted because it is a fact about the machine.
"""

from __future__ import annotations

import torch

from qd_train.run_control import CostEstimate


def n_gpus_for_device(device: str) -> int | None:
    """How many GPUs this run is being billed for, counted rather than assumed.

    ``None`` for a local device, where :meth:`CostEstimate.for_device` prices at zero. On
    cuda it is the visible device count -- which is what a price list charges for, and what
    gates both the per-GPU column check and the multi-GPU half of rule 4.

    Any other device name returns ``None``, which is not a default: it reaches
    :meth:`CostEstimate.for_device` as a missing value and is refused there by name,
    together with whichever of ``instance`` and ``usd_per_hour`` is also missing. A number
    invented here would be the quieter answer and the wrong one.
    """
    if device in CostEstimate.LOCAL_DEVICES:
        return None
    return torch.cuda.device_count() if device == "cuda" else None
