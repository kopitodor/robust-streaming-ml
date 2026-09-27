"""Configuration schema and loading.

Configurations are YAML files validated against a strict pydantic schema:
unknown keys, wrong types, and out-of-range values are rejected rather than
silently ignored. Later milestones extend :class:`ExperimentConfig` with
dataset, feature, model, and evaluation sections.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "ConfigError",
    "ExperimentConfig",
    "RunConfig",
    "load_config",
    "parse_config",
]

# Seeds must be accepted by random, numpy.random.seed, and torch.manual_seed.
MAX_SEED = 2**32 - 1
_NAME_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"


class ConfigError(ValueError):
    """Raised when a configuration file cannot be read or fails validation."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class RunConfig(_StrictModel):
    """Where and how run artifacts are written."""

    output_root: Path = Field(
        default=Path("runs"),
        # YAML yields strings; this is the one field allowed to coerce.
        strict=False,
        description="Directory under which per-run directories are created.",
    )


class ExperimentConfig(_StrictModel):
    """Top-level experiment configuration."""

    name: str = Field(
        min_length=1,
        max_length=64,
        pattern=_NAME_PATTERN,
        description="Filesystem-safe experiment name, used in run-directory names.",
    )
    seed: int = Field(ge=0, le=MAX_SEED, description="Global random seed.")
    deterministic: bool = Field(
        default=True,
        description="Request deterministic PyTorch algorithms.",
    )
    run: RunConfig = Field(default_factory=RunConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON/YAML-serializable representation."""
        return self.model_dump(mode="json")


def parse_config(data: Any, *, source: str = "<config>") -> ExperimentConfig:
    """Validate an already-parsed mapping as an :class:`ExperimentConfig`."""
    if not isinstance(data, dict):
        raise ConfigError(
            f"{source}: top-level configuration must be a mapping, got {type(data).__name__}"
        )
    try:
        return ExperimentConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"{source}: invalid configuration\n{exc}") from exc


def load_config(path: str | Path) -> ExperimentConfig:
    """Load and validate a YAML configuration file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{path}: cannot read configuration: {exc}") from exc
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path}: invalid YAML: {exc}") from exc
    return parse_config(data, source=str(path))
