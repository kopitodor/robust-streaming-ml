"""Baseline keyword-spotting CNN (``kws_cnn_v1``).

Input: log-mel features ``[B, 1, n_mels, frames]`` (``[B, 1, 40, 101]`` by
default). Output: unnormalised class logits ``[B, num_classes]``.

Layers:

1. ``input_norm``: ``BatchNorm2d(1)``, which learns the scale and offset of
   the log-mel input (log values are otherwise roughly -14 to +5);
2. for each entry ``c`` of ``channels``: ``Conv2d(k=3, padding=1, bias=False)``
   -> ``BatchNorm2d(c)`` -> ``ReLU`` -> ``MaxPool2d(2)`` (halves both axes,
   flooring odd sizes);
3. ``AdaptiveAvgPool2d(1)`` (global average over time and frequency), so the
   classifier does not depend on the exact frame count;
4. ``Dropout(p)`` -> ``Linear(channels[-1], num_classes)``.

With the default ``channels = [16, 32, 64, 64]``, 40 mels, 101 frames, and 10
classes, the shapes are ``[B,1,40,101] -> [B,16,20,50] -> [B,32,10,25] ->
[B,64,5,12] -> [B,64,2,6] -> [B,64] -> [B,10]`` with 61,052 trainable
parameters (see :func:`count_parameters`; a test pins this number).
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from robust_streaming_ml.training.config import ModelSection

__all__ = ["KeywordCNN", "count_parameters", "layer_shapes"]


class KeywordCNN(nn.Module):
    def __init__(self, config: ModelSection, num_classes: int, n_mels: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        pools = len(config.channels)
        if n_mels < 2**pools:
            raise ValueError(f"n_mels={n_mels} is too small for {pools} 2x2 pooling stages")
        self.config = config
        self.num_classes = num_classes
        self.input_norm = nn.BatchNorm2d(1)
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in config.channels:
            blocks += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
            in_channels = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 4 or features.shape[1] != 1:
            raise ValueError(f"expected [batch, 1, n_mels, frames], got {list(features.shape)}")
        x = self.blocks(self.input_norm(features))
        x = torch.flatten(self.pool(x), 1)
        logits: torch.Tensor = self.classifier(self.dropout(x))
        return logits


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters (batch-norm running statistics excluded)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_shapes(model: KeywordCNN, input_shape: tuple[int, int, int]) -> list[dict[str, Any]]:
    """Output shape (batch dimension omitted) after the input norm, each block, and the head."""
    shapes: list[dict[str, Any]] = [{"layer": "input", "shape": list(input_shape)}]
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = model.input_norm(torch.zeros(1, *input_shape))
            for index, module in enumerate(model.blocks):
                x = module(x)
                if isinstance(module, nn.MaxPool2d):
                    shapes.append({"layer": f"block{index // 4 + 1}", "shape": list(x.shape[1:])})
            x = torch.flatten(model.pool(x), 1)
            shapes.append({"layer": "global_pool", "shape": list(x.shape[1:])})
            x = model.classifier(model.dropout(x))
            shapes.append({"layer": "logits", "shape": list(x.shape[1:])})
    finally:
        model.train(was_training)
    return shapes
