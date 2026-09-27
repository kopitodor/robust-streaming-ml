"""Robustness configuration (``schema_version`` 1).

The protocol itself (corruptions, versions, severities, parameters, seeds,
metrics) is fixed in code under ``kw10-robustness-v1`` and documented in
docs/robustness-protocol.md. The config names the inputs: the validated
manifest, the pinned checkpoint, and the pinned Milestone 4 evaluation run
that provides the clean baseline. It also sets runtime options that do not
change the definitions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

from pydantic import Field

from robust_streaming_ml.config import ExperimentConfig, StrictModel
from robust_streaming_ml.evaluation.config import CheckpointRef
from robust_streaming_ml.training.config import DataSection

__all__ = ["BaselineRef", "RobustnessConfig", "RobustnessSection"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BaselineRef(StrictModel):
    """The Milestone 4 evaluation run whose official raw test metrics are the clean baseline."""

    run_dir: Path = Field(strict=False)
    report_sha256: str = Field(pattern=_SHA256.pattern)


class RobustnessSection(StrictModel):
    protocol: Literal["kw10-robustness-v1"] = "kw10-robustness-v1"
    batch_size: int = Field(default=256, ge=1, le=16384)
    num_threads: int | None = Field(default=None, ge=1, le=256)
    corruption_workers: int = Field(default=8, ge=1, le=64)
    retain_audio_examples: int = Field(
        default=0, ge=0, le=100, description="Corrupted clips written as float32 WAV per condition."
    )


class RobustnessConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    data: DataSection
    checkpoint: CheckpointRef
    baseline: BaselineRef
    robustness: RobustnessSection = Field(default_factory=RobustnessSection)
