"""Export and parity configuration (``schema_version`` 1).

The streaming contract (``kw10-streaming-v1``) and the export and parity
protocol (``kw10-onnx-v1``: opset, tensor contract, runtime, session options,
parity sets, tolerances, chunk seeds) are fixed in code and documented in
docs/streaming-onnx-protocol.md. The config names the inputs (the validated
manifest, the pinned checkpoint, the pinned Milestone 4 evaluation run) and
runtime options that do not change the definitions.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from robust_streaming_ml.config import ExperimentConfig, StrictModel
from robust_streaming_ml.evaluation.config import CheckpointRef
from robust_streaming_ml.robustness.config import BaselineRef
from robust_streaming_ml.training.config import DataSection

__all__ = ["ExportConfig", "ExportSection"]


class ExportSection(StrictModel):
    protocol: Literal["kw10-onnx-v1"] = "kw10-onnx-v1"
    streaming_protocol: Literal["kw10-streaming-v1"] = "kw10-streaming-v1"
    reference_batch_size: int = Field(
        default=256, ge=1, le=16384, description="Batch size of the clean-path gate."
    )
    num_threads: int | None = Field(default=None, ge=1, le=256, description="torch threads.")


class ExportConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    data: DataSection
    checkpoint: CheckpointRef
    baseline: BaselineRef
    export: ExportSection = Field(default_factory=ExportSection)
