from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch

from robust_streaming_ml.config import MAX_SEED
from robust_streaming_ml.seed import (
    SeedState,
    numpy_generator,
    seed_everything,
    torch_generator,
)


def draw() -> tuple[list[float], np.ndarray, torch.Tensor]:
    return (
        [random.random() for _ in range(5)],
        np.random.rand(5),  # noqa: NPY002 - exercising the legacy global RNG
        torch.rand(5),
    )


def assert_draws_equal(
    a: tuple[list[float], np.ndarray, torch.Tensor],
    b: tuple[list[float], np.ndarray, torch.Tensor],
) -> None:
    assert a[0] == b[0]
    np.testing.assert_array_equal(a[1], b[1])
    assert torch.equal(a[2], b[2])


def test_same_seed_reproduces_all_global_rngs() -> None:
    seed_everything(123)
    first = draw()
    seed_everything(123)
    assert_draws_equal(first, draw())


def test_different_seeds_differ() -> None:
    seed_everything(1)
    first = draw()
    seed_everything(2)
    second = draw()
    assert first[0] != second[0]
    assert not np.array_equal(first[1], second[1])
    assert not torch.equal(first[2], second[2])


def test_reseeding_resets_mid_sequence() -> None:
    seed_everything(7)
    expected = draw()
    seed_everything(7)
    draw()  # advance every generator
    seed_everything(7)
    assert_draws_equal(expected, draw())


def test_seeded_model_initialization_is_reproducible() -> None:
    seed_everything(99)
    first = torch.nn.Linear(8, 4).state_dict()
    seed_everything(99)
    second = torch.nn.Linear(8, 4).state_dict()
    for key, value in first.items():
        assert torch.equal(value, second[key])


def test_deterministic_mode_configures_torch() -> None:
    state = seed_everything(5, deterministic=True)
    assert state == SeedState(seed=5, deterministic=True)
    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    assert os.environ["PYTHONHASHSEED"] == "5"
    assert os.environ["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"


def test_non_deterministic_mode_configures_torch() -> None:
    state = seed_everything(5, deterministic=False)
    assert state.deterministic is False
    assert not torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True


def test_explicit_generators_are_reproducible_and_isolated() -> None:
    seed_everything(0)
    np_a = numpy_generator(42).random(5)
    torch_a = torch.rand(5, generator=torch_generator(42))
    seed_everything(1)  # changing global state must not affect explicit generators
    np_b = numpy_generator(42).random(5)
    torch_b = torch.rand(5, generator=torch_generator(42))
    np.testing.assert_array_equal(np_a, np_b)
    assert torch.equal(torch_a, torch_b)


def test_explicit_generators_do_not_consume_global_state() -> None:
    seed_everything(3)
    expected = draw()
    seed_everything(3)
    numpy_generator(3).random(10)
    torch.rand(10, generator=torch_generator(3))
    assert_draws_equal(expected, draw())


@pytest.mark.parametrize("seed", [0, MAX_SEED])
def test_seed_bounds_accepted(seed: int) -> None:
    assert seed_everything(seed).seed == seed
    numpy_generator(seed)
    torch_generator(seed)


@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1])
def test_out_of_range_seed_rejected(seed: int) -> None:
    with pytest.raises(ValueError, match="seed must be in"):
        seed_everything(seed)
    with pytest.raises(ValueError, match="seed must be in"):
        numpy_generator(seed)
    with pytest.raises(ValueError, match="seed must be in"):
        torch_generator(seed)


@pytest.mark.parametrize("seed", [1.0, "1", True, None])
def test_non_int_seed_rejected(seed: object) -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        seed_everything(seed)  # type: ignore[arg-type]
