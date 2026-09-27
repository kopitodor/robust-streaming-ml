"""Structured, self-describing run directories.

Each run gets a fresh directory ``<output_root>/<run_id>/`` containing:

- ``config.yaml``   -- the validated configuration exactly as used;
- ``metadata.json`` -- run identity, config hash, and environment metadata
  (Python, platform, key package versions, PyTorch runtime, git revision).

Run directories are never reused: creating one that already exists fails.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import torch
import yaml

from robust_streaming_ml import __version__
from robust_streaming_ml.config import ExperimentConfig

__all__ = [
    "CONFIG_FILENAME",
    "METADATA_FILENAME",
    "RunDirectory",
    "collect_environment_metadata",
    "config_digest",
    "create_run_dir",
]

CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 1

_TRACKED_PACKAGES = ("numpy", "pydantic", "pyyaml", "torch")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class RunDirectory:
    """A created run directory and the metadata written into it."""

    path: Path
    run_id: str
    metadata: dict[str, Any]

    @property
    def config_path(self) -> Path:
        return self.path / CONFIG_FILENAME

    @property
    def metadata_path(self) -> Path:
        return self.path / METADATA_FILENAME


def config_digest(config: ExperimentConfig) -> str:
    """SHA-256 of the canonical JSON form of ``config``."""
    canonical = json.dumps(config.to_dict(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def _git_info(cwd: Path) -> dict[str, Any] | None:
    """Commit and dirty state of the repository containing ``cwd``, if any."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        ).stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except (OSError, subprocess.SubprocessError):
        return None
    return {"commit": commit, "dirty": dirty}


def collect_environment_metadata(cwd: Path | None = None) -> dict[str, Any]:
    """Describe the interpreter, platform, packages, and repository revision."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "packages": {
            "robust-streaming-ml": __version__,
            **{name: _package_version(name) for name in _TRACKED_PACKAGES},
        },
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),  # type: ignore[no-untyped-call]
            "num_threads": torch.get_num_threads(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "git": _git_info(cwd if cwd is not None else Path.cwd()),
        "argv": list(sys.argv),
    }


def _default_run_id(config: ExperimentConfig, created_at: datetime) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{config.name}_{uuid.uuid4().hex[:8]}"


def create_run_dir(
    config: ExperimentConfig,
    *,
    root: Path | None = None,
    run_id: str | None = None,
) -> RunDirectory:
    """Create a new run directory recording ``config`` and environment metadata.

    ``root`` defaults to ``config.run.output_root``. ``run_id`` defaults to
    ``<UTC timestamp>_<config name>_<random suffix>``. Raises
    ``FileExistsError`` if the target directory already exists.
    """
    created_at = datetime.now(timezone.utc)
    if run_id is None:
        run_id = _default_run_id(config, created_at)
    elif not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run_id must be a filesystem-safe name, got {run_id!r}")

    base = root if root is not None else config.run.output_root
    path = base / run_id
    base.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)

    run_metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "experiment_name": config.name,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "config_file": CONFIG_FILENAME,
        "config_sha256": config_digest(config),
        "environment": collect_environment_metadata(),
    }

    (path / CONFIG_FILENAME).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8"
    )
    (path / METADATA_FILENAME).write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunDirectory(path=path, run_id=run_id, metadata=run_metadata)
