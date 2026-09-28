"""Demo configuration (``schema_version`` 1).

The demo runs the pinned Milestone 6 export (its report must record passed
parity) under the fixed ``kw10-streaming-v1`` contract. The config names that
export and a few presentation options; it cannot change the model, the
streaming contract, or the corruption definitions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from robust_streaming_ml.config import ExperimentConfig, StrictModel
from robust_streaming_ml.export.config import ExportRef

__all__ = ["DemoConfig", "DemoSection"]


class DemoSection(StrictModel):
    output_dir: Path = Field(
        default=Path("artifacts/demo"), strict=False, description="Git-ignored output folder."
    )
    max_seconds: float = Field(default=60.0, gt=0.0, le=600.0, description="Longest input.")
    intra_op_threads: int = Field(default=1, ge=1, le=64)
    embed_audio: bool = Field(default=True, description="Embed playable audio in the page.")


class DemoConfig(ExperimentConfig):
    schema_version: Literal[1] = 1
    export: ExportRef
    demo: DemoSection = Field(default_factory=DemoSection)
