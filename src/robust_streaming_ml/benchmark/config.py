"""Benchmark configuration (``schema_version`` 1).

The protocol (paths, thread configurations, warm-up, timed iterations,
repetitions, timer, and percentile method) is fixed in code as
``kw10-latency-v1`` and documented in docs/benchmark-protocol.md. The config
names the inputs: the validated manifest (for the benchmark audio), the
pinned checkpoint (the PyTorch reference path), and the pinned Milestone 6
export run.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from robust_streaming_ml.config import ExperimentConfig, StrictModel
from robust_streaming_ml.evaluation.config import CheckpointRef
from robust_streaming_ml.export.config import ExportRef
from robust_streaming_ml.training.config import DataSection

__all__ = ["BenchmarkConfig", "BenchmarkSection"]


class BenchmarkSection(StrictModel):
    protocol: Literal["kw10-latency-v1"] = "kw10-latency-v1"


class BenchmarkConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    data: DataSection
    checkpoint: CheckpointRef
    export: ExportRef
    benchmark: BenchmarkSection = Field(default_factory=BenchmarkSection)
