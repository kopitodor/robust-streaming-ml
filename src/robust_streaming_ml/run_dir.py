"""Run start: provenance, seeding, and self-describing run directories.

:func:`start_run` is the single entry point for beginning a run. In order, it

1. resolves the source revision (git commit and dirty state) of the code
   being run -- always from this package's own directory, never from a
   caller-chosen path -- failing with :class:`ProvenanceError` if git cannot
   provide it or the package files are not tracked by that repository;
2. applies seeding with :func:`~robust_streaming_ml.seed.seed_everything` and
   verifies the resulting process state;
3. creates a fresh directory ``<output_root>/<run_id>/`` containing
   ``config.yaml`` (the validated configuration) and ``metadata.json`` (run
   identity, config hash, source revision, applied seed state, environment).

There is no revision override: a run whose code revision cannot be resolved
is not started. A non-editable install (e.g. a wheel in site-packages) has no
repository revision, so runs must use an editable install of a git checkout.
Run directories are never reused.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from robust_streaming_ml.seed import seed_everything

__all__ = [
    "CONFIG_FILENAME",
    "METADATA_FILENAME",
    "ProvenanceError",
    "RunDirectory",
    "SourceRevision",
    "collect_environment_metadata",
    "config_digest",
    "resolve_source_revision",
    "start_run",
]

CONFIG_FILENAME = "config.yaml"
METADATA_FILENAME = "metadata.json"
METADATA_SCHEMA_VERSION = 2

# Directory whose repository revision identifies the code being run. Tests
# may monkeypatch it; there is deliberately no public parameter to override it.
PACKAGE_DIR = Path(__file__).resolve().parent

_GIT_EXECUTABLE = "git"
_TRACKED_PACKAGES = ("numpy", "pydantic", "pyyaml", "torch")
_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}([0-9a-f]{24})?$")


class ProvenanceError(RuntimeError):
    """Raised when the source revision of a run cannot be established."""


@dataclass(frozen=True)
class SourceRevision:
    """Git revision of the code being run."""

    commit: str
    dirty: bool


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


def resolve_source_revision(source_dir: Path = PACKAGE_DIR) -> SourceRevision:
    """Return the commit and dirty state of the repository containing ``source_dir``.

    ``source_dir`` must contain files tracked by that repository; otherwise the
    commit would not identify its code (for example, a package installed into
    an ignored virtual environment inside some checkout). ``dirty`` is true if
    any tracked file is modified or any untracked, non-ignored file exists.
    Raises :class:`ProvenanceError`, including git's diagnostic, if the
    revision cannot be resolved.
    """

    def git(*args: str) -> str:
        try:
            completed = subprocess.run(
                [_GIT_EXECUTABLE, *args],
                cwd=source_dir,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip() or f"exit status {exc.returncode}"
            raise ProvenanceError(
                f"cannot resolve source revision in {source_dir}: "
                f"`git {' '.join(args)}` failed:\n{detail}"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise ProvenanceError(
                f"cannot resolve source revision in {source_dir}: cannot run git: {exc}"
            ) from exc
        return completed.stdout

    commit = git("rev-parse", "--verify", "HEAD^{commit}").strip()
    if not _COMMIT_PATTERN.fullmatch(commit):
        raise ProvenanceError(f"git returned an unexpected commit id: {commit!r}")
    if not git("ls-files", "--", ".").strip():
        raise ProvenanceError(
            f"cannot resolve source revision in {source_dir}: it contains no files "
            "tracked by the enclosing git repository, so that repository's commit "
            "does not identify this code (use an editable install of a git checkout)"
        )
    dirty = bool(git("status", "--porcelain").strip())
    return SourceRevision(commit=commit, dirty=dirty)


def _package_version(name: str) -> str | None:
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_environment_metadata() -> dict[str, Any]:
    """Describe the interpreter, platform, packages, and PyTorch runtime."""
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
        },
        "argv": list(sys.argv),
    }


def _apply_seeding(config: ExperimentConfig) -> dict[str, Any]:
    """Seed the process and return the state observed afterwards."""
    state = seed_everything(config.seed, deterministic=config.deterministic)
    observed = {
        "seed": state.seed,
        "deterministic": state.deterministic,
        "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = {
        "seed": config.seed,
        "deterministic": config.deterministic,
        "torch_deterministic_algorithms": config.deterministic,
        "cudnn_deterministic": config.deterministic,
        "cudnn_benchmark": not config.deterministic,
        "PYTHONHASHSEED": str(config.seed),
    }
    mismatched = {k: observed[k] for k, v in expected.items() if observed[k] != v}
    if mismatched:
        raise RuntimeError(f"seeding did not take effect as configured: {mismatched}")
    return observed


def _default_run_id(config: ExperimentConfig, created_at: datetime) -> str:
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{config.name}_{uuid.uuid4().hex[:8]}"


def start_run(
    config: ExperimentConfig,
    *,
    root: Path | None = None,
    run_id: str | None = None,
) -> RunDirectory:
    """Start a run: resolve provenance, apply seeding, and create its directory.

    ``root`` defaults to ``config.run.output_root``. ``run_id`` defaults to
    ``<UTC timestamp>_<config name>_<random suffix>``. The recorded revision
    is always that of the repository containing this package's source.

    Raises :class:`ProvenanceError` (before seeding or creating anything) if
    the source revision cannot be resolved, and ``FileExistsError`` if the
    run directory already exists.
    """
    created_at = datetime.now(timezone.utc)
    if run_id is None:
        run_id = _default_run_id(config, created_at)
    elif not _RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(f"run_id must be a filesystem-safe name, got {run_id!r}")

    base = root if root is not None else config.run.output_root
    path = base / run_id
    if path.exists():
        raise FileExistsError(f"run directory already exists: {path}")

    source = resolve_source_revision(PACKAGE_DIR)
    seeding = _apply_seeding(config)
    run_metadata: dict[str, Any] = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at_utc": created_at.isoformat(),
        "experiment_name": config.name,
        "config_file": CONFIG_FILENAME,
        "config_sha256": config_digest(config),
        "source": {"commit": source.commit, "dirty": source.dirty},
        "seeding": seeding,
        "environment": collect_environment_metadata(),
    }

    base.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    (path / CONFIG_FILENAME).write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=True), encoding="utf-8"
    )
    (path / METADATA_FILENAME).write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunDirectory(path=path, run_id=run_id, metadata=run_metadata)
