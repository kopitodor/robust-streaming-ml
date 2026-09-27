from __future__ import annotations

import pytest
import torch

from robust_streaming_ml.model import KeywordCNN, count_parameters, layer_shapes
from robust_streaming_ml.training.config import ModelSection


def test_default_parameter_count_is_pinned() -> None:
    # input BN 2; convs 144 + 4608 + 18432 + 36864; block BNs 32 + 64 + 128 + 128; linear 650.
    assert count_parameters(KeywordCNN(ModelSection(), num_classes=10, n_mels=40)) == 61052


def test_default_layer_shapes() -> None:
    model = KeywordCNN(ModelSection(), num_classes=10, n_mels=40)
    assert layer_shapes(model, (1, 40, 101)) == [
        {"layer": "input", "shape": [1, 40, 101]},
        {"layer": "block1", "shape": [16, 20, 50]},
        {"layer": "block2", "shape": [32, 10, 25]},
        {"layer": "block3", "shape": [64, 5, 12]},
        {"layer": "block4", "shape": [64, 2, 6]},
        {"layer": "global_pool", "shape": [64]},
        {"layer": "logits", "shape": [10]},
    ]
    assert model.training  # layer_shapes restores the mode


@pytest.mark.parametrize(
    ("channels", "classes", "input_shape"),
    [([16, 32, 64, 64], 10, (1, 40, 101)), ([4, 8], 2, (1, 16, 101)), ([8], 3, (1, 40, 51))],
)
def test_forward_output_shape(
    channels: list[int], classes: int, input_shape: tuple[int, ...]
) -> None:
    model = KeywordCNN(ModelSection(channels=channels), num_classes=classes, n_mels=input_shape[1])
    logits = model(torch.randn(5, *input_shape))
    assert logits.shape == (5, classes)
    assert logits.dtype == torch.float32


def test_parameter_count_formula() -> None:
    channels, classes = [6, 10], 4
    model = KeywordCNN(ModelSection(channels=channels), num_classes=classes, n_mels=8)
    expected = 2 + (1 * 6 * 9 + 2 * 6) + (6 * 10 * 9 + 2 * 10) + (10 * classes + classes)
    assert count_parameters(model) == expected


def test_initialisation_is_seeded() -> None:
    torch.manual_seed(3)
    a = KeywordCNN(ModelSection(), 10, 40).state_dict()
    torch.manual_seed(3)
    b = KeywordCNN(ModelSection(), 10, 40).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_eval_mode_is_deterministic_and_gradients_reach_every_parameter() -> None:
    model = KeywordCNN(ModelSection(), 10, 40)
    x = torch.randn(4, 1, 40, 101)
    loss = torch.nn.functional.cross_entropy(model(x), torch.tensor([0, 1, 2, 3]))
    loss.backward()  # type: ignore[no-untyped-call]
    assert all(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())
    model.eval()
    assert torch.equal(model(x), model(x))


def test_invalid_inputs_and_settings_are_rejected() -> None:
    model = KeywordCNN(ModelSection(), 10, 40)
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(2, 40, 101))
    with pytest.raises(ValueError, match="expected"):
        model(torch.zeros(2, 2, 40, 101))
    with pytest.raises(ValueError, match="too small"):
        KeywordCNN(ModelSection(), 10, n_mels=8)
    with pytest.raises(ValueError, match="num_classes"):
        KeywordCNN(ModelSection(), 1, 40)
    with pytest.raises(ValueError):
        ModelSection.model_validate({"channels": [0]})
    with pytest.raises(ValueError):
        ModelSection.model_validate({"architecture": "resnet"})
    with pytest.raises(ValueError):
        ModelSection.model_validate({"dropout": 1.0})
