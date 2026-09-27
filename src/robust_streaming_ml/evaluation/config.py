"""Evaluation configuration (``schema_version`` 1).

The protocol's fixed choices (splits, ECE bins, calibration method and search)
are constants of ``kw10-clean-eval-v1`` in code, not configurable, so a config
cannot silently change the protocol. The config names the protocol, the
validated manifest (the same ``data`` section as training), the pinned
checkpoint, and runtime settings that do not change the definitions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from robust_streaming_ml.config import ExperimentConfig, StrictModel
from robust_streaming_ml.training.config import DataSection

__all__ = ["ECE_BINS", "PROTOCOL", "CheckpointRef", "EvaluationConfig", "EvaluationSection"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")

ECE_BINS = 15
PROTOCOL: dict[str, object] = {
    "id": "kw10-clean-eval-v1",
    "splits": ["validation", "test"],
    "ece_bins": ECE_BINS,
    "ece_binning": "equal-width [k/B, (k+1)/B), last bin includes 1.0; top-label confidence",
    "macro_f1_zero_division": 0.0,
    "prediction": "argmax of logits, ties to lowest label index",
    "calibration": "temperature scaling fitted on validation NLL only",
    "duplicates": "official: all rows; secondary: one row per audio_sha256 (smallest path)",
}


class CheckpointRef(StrictModel):
    """The checkpoint under evaluation, pinned by file SHA-256."""

    path: Path = Field(strict=False)
    sha256: str = Field(pattern=_SHA256.pattern)


class EvaluationSection(StrictModel):
    protocol: Literal["kw10-clean-eval-v1"] = "kw10-clean-eval-v1"
    batch_size: int = Field(default=256, ge=1, le=16384)
    num_threads: int | None = Field(default=None, ge=1, le=256)


class EvaluationConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    data: DataSection
    checkpoint: CheckpointRef
    evaluation: EvaluationSection = Field(default_factory=EvaluationSection)
