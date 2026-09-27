"""Deterministic seeding for Python, NumPy, and PyTorch.

``seed_everything`` seeds the process-global generators. Prefer the explicit
generators from ``numpy_generator``/``torch_generator`` in new code so that
randomness is passed around rather than drawn from global state.

Determinism limits: ``PYTHONHASHSEED`` only affects interpreters started after
it is set (for example worker subprocesses), and deterministic PyTorch mode
makes operations without a deterministic implementation raise instead of
silently varying.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from robust_streaming_ml.config import MAX_SEED

__all__ = [
    "SeedState",
    "numpy_generator",
    "seed_everything",
    "torch_generator",
]

# Required by cuBLAS for deterministic results on CUDA >= 10.2.
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"


@dataclass(frozen=True)
class SeedState:
    """What ``seed_everything`` configured; suitable for run metadata."""

    seed: int
    deterministic: bool


def _check_seed(seed: int) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, {MAX_SEED}], got {seed}")


def seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
    """Seed Python, NumPy, and PyTorch global RNGs.

    With ``deterministic=True`` PyTorch is also configured to use deterministic
    algorithms and cuDNN autotuning is disabled.
    """
    _check_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeding legacy global state is the point
    torch.manual_seed(seed)  # also seeds all CUDA devices

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", _CUBLAS_WORKSPACE_CONFIG)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic
    return SeedState(seed=seed, deterministic=deterministic)


def numpy_generator(seed: int) -> np.random.Generator:
    """Return an independent NumPy generator seeded with ``seed``."""
    _check_seed(seed)
    return np.random.default_rng(seed)


def torch_generator(seed: int, device: str | torch.device = "cpu") -> torch.Generator:
    """Return an independent PyTorch generator seeded with ``seed``."""
    _check_seed(seed)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
