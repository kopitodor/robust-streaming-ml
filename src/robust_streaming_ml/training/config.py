"""Versioned configuration for baseline training.

:class:`TrainingConfig` extends :class:`~robust_streaming_ml.config.ExperimentConfig`
(so ``start_run`` records and hashes every section) with:

- ``data``: the validated manifest, its pinned SHA-256, the validation report,
  the audio root, and the label order;
- ``window``: the fixed audio window policy (versioned);
- ``features``: log-mel extraction settings (versioned);
- ``model``: the CNN architecture;
- ``training``: optimisation, batching, validation, workers, and augmentation;
- ``checkpoint``: which checkpoints are written.

All sections are strict: unknown keys, wrong types, and inconsistent values
fail validation. ``schema_version``, ``window.version``, and
``features.version`` change whenever their meaning changes; checkpoints
record them and are rejected when they do not match.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from robust_streaming_ml.config import ExperimentConfig, StrictModel

__all__ = [
    "AugmentationSection",
    "CheckpointSection",
    "DataSection",
    "FeatureSection",
    "ModelSection",
    "TrainingConfig",
    "TrainingSection",
    "WindowSection",
]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class DataSection(StrictModel):
    """Inputs: only a validated, pinned manifest may be used."""

    dataset: str = Field(min_length=1, description="Dataset spec name (manifest_name).")
    manifest_path: Path = Field(strict=False)
    manifest_sha256: str = Field(pattern=_SHA256.pattern)
    validation_report_path: Path = Field(strict=False)
    audio_root: Path = Field(strict=False, description="Extraction root the manifest refers to.")
    labels: list[str] = Field(min_length=2, description="Class order; index = label_index.")
    load_workers: int = Field(default=8, ge=1, le=64, description="Threads for reading audio.")

    @model_validator(mode="after")
    def _unique_labels(self) -> DataSection:
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        return self


class WindowSection(StrictModel):
    """Fixed-length analysis window applied to every clip.

    ``start_zero_pad`` keeps the clip at sample 0 and appends zeros up to
    ``num_samples``. A clip longer than the window is an error: nothing is
    cropped, repaired, or resampled.
    """

    version: Literal[1] = 1
    sample_rate_hz: int = Field(default=16000, ge=1)
    num_samples: int = Field(default=16000, ge=1)
    alignment: Literal["start_zero_pad"] = "start_zero_pad"


class FeatureSection(StrictModel):
    """Log-mel spectrogram settings (see ``robust_streaming_ml.features``)."""

    version: Literal[1] = 1
    kind: Literal["log_mel"] = "log_mel"
    n_fft: int = Field(default=512, ge=16)
    win_length: int = Field(default=400, ge=16)
    hop_length: int = Field(default=160, ge=1)
    n_mels: int = Field(default=40, ge=1)
    f_min_hz: float = Field(default=20.0, ge=0.0)
    f_max_hz: float = Field(default=8000.0, gt=0.0)
    log_offset: float = Field(default=1.0e-6, gt=0.0)

    @model_validator(mode="after")
    def _consistent(self) -> FeatureSection:
        if self.win_length > self.n_fft:
            raise ValueError("win_length must not exceed n_fft")
        if self.hop_length > self.win_length:
            raise ValueError("hop_length must not exceed win_length")
        if self.f_min_hz >= self.f_max_hz:
            raise ValueError("f_min_hz must be below f_max_hz")
        if self.n_mels > self.n_fft // 2 + 1:
            raise ValueError("n_mels must not exceed the number of FFT bins")
        return self


class ModelSection(StrictModel):
    """``kws_cnn_v1``: conv blocks (3x3 conv, batch norm, ReLU, 2x2 max pool),
    global average pooling, dropout, and one linear layer."""

    architecture: Literal["kws_cnn_v1"] = "kws_cnn_v1"
    channels: list[int] = Field(default=[16, 32, 64, 64], min_length=1, max_length=6)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _positive_channels(self) -> ModelSection:
        if any(c < 1 or c > 512 for c in self.channels):
            raise ValueError("channels must be between 1 and 512")
        return self


class AugmentationSection(StrictModel):
    """Train-only augmentation. Disabled unless explicitly enabled.

    ``time_shift``: shift the windowed waveform by an integer number of samples
    drawn uniformly from ``[-max, +max]`` (``max = time_shift_max_ms`` in
    samples), filling with zeros. The shift for sample ``i`` in epoch ``e`` comes
    from ``derive_seed(seed, "augment", e, i)``, so it does not depend on the
    number of DataLoader workers.
    """

    enabled: bool = False
    time_shift_max_ms: float = Field(default=100.0, gt=0.0, le=500.0)


class TrainingSection(StrictModel):
    epochs: int = Field(ge=1, le=1000)
    batch_size: int = Field(default=64, ge=1, le=4096)
    validation_batch_size: int = Field(default=256, ge=1, le=16384)
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.0001, ge=0.0, le=1.0)
    lr_schedule: Literal["constant", "cosine"] = "cosine"
    selection_metric: Literal["validation_loss"] = "validation_loss"
    num_workers: int = Field(default=0, ge=0, le=32, description="DataLoader worker processes.")
    num_threads: int | None = Field(default=None, ge=1, le=256, description="torch threads.")
    augmentation: AugmentationSection = Field(default_factory=AugmentationSection)


class CheckpointSection(StrictModel):
    """``best`` (lowest validation loss) is always written; ``last`` optionally."""

    save_last: bool = True


class TrainingConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    data: DataSection
    window: WindowSection = Field(default_factory=WindowSection)
    features: FeatureSection = Field(default_factory=FeatureSection)
    model: ModelSection = Field(default_factory=ModelSection)
    training: TrainingSection
    checkpoint: CheckpointSection = Field(default_factory=CheckpointSection)

    @model_validator(mode="after")
    def _cross_section(self) -> TrainingConfig:
        if self.features.f_max_hz > self.window.sample_rate_hz / 2:
            raise ValueError("features.f_max_hz must not exceed the Nyquist frequency")
        if self.features.win_length > self.window.num_samples:
            raise ValueError("features.win_length must not exceed window.num_samples")
        return self
