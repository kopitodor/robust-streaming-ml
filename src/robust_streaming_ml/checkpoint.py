"""Versioned, inspectable model checkpoints (format version 1).

A checkpoint is two files:

- ``<name>.pt``: ``torch.save`` of a plain dict
  ``{"format", "format_version", "metadata", "model_state"}``. It contains only
  tensors and JSON-compatible values, so it loads with ``weights_only=True``
  (no arbitrary code is unpickled);
- ``<name>.json``: the same metadata plus the ``.pt`` file's SHA-256 and the
  name, shape, and dtype of every tensor, readable without PyTorch.

``metadata`` records everything needed to reuse the weights safely: the full
config and its digest, labels, model architecture and parameter count,
window/feature settings with the feature version and fingerprint, manifest
identity (dataset, SHA-256, rows per split), source revision, seeding (applied
state and derived seeds), run id, epoch, and validation monitoring values.

:func:`load_checkpoint` fails closed (:class:`CheckpointError`) on:
- a wrong format or version, missing metadata, or a SHA-256 mismatch with the
  sidecar;
- an unsupported architecture or a feature version or fingerprint that differs
  from this code's;
- a state dict that does not load strictly;
- a mismatch with any expectation the caller passes (manifest SHA-256, labels,
  window, features).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from pydantic import ValidationError

from robust_streaming_ml.features import FEATURE_VERSION, LogMelSpectrogram, feature_fingerprint
from robust_streaming_ml.model import KeywordCNN, count_parameters
from robust_streaming_ml.training.config import FeatureSection, ModelSection, WindowSection

__all__ = [
    "CHECKPOINT_FORMAT",
    "CHECKPOINT_FORMAT_VERSION",
    "REQUIRED_METADATA",
    "CheckpointError",
    "LoadedCheckpoint",
    "load_checkpoint",
    "save_checkpoint",
    "sidecar_path",
]

CHECKPOINT_FORMAT = "robust-streaming-ml/checkpoint"
CHECKPOINT_FORMAT_VERSION = 1
REQUIRED_METADATA: dict[str, tuple[str, ...]] = {
    "kind": (),
    "epoch": (),
    "run_id": (),
    "labels": (),
    "config": (),
    "config_sha256": (),
    "model": ("architecture", "config", "num_classes", "num_parameters", "input_shape"),
    "preprocessing": ("window", "features", "feature_version", "feature_fingerprint"),
    "manifest": ("dataset", "sha256", "rows_by_split"),
    "source": ("commit", "dirty"),
    "seeding": (),
    "validation": ("loss", "accuracy"),
    "torch_version": (),
}
_SUPPORTED_ARCHITECTURES = ("kws_cnn_v1",)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint is malformed or incompatible."""


@dataclass(frozen=True)
class LoadedCheckpoint:
    model: KeywordCNN
    frontend: LogMelSpectrogram
    metadata: dict[str, Any]
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metadata(metadata: Any) -> None:
    if not isinstance(metadata, dict):
        raise CheckpointError("checkpoint metadata must be a mapping")
    for key, subkeys in REQUIRED_METADATA.items():
        if key not in metadata:
            raise CheckpointError(f"checkpoint metadata is missing {key!r}")
        for sub in subkeys:
            if not isinstance(metadata[key], dict) or sub not in metadata[key]:
                raise CheckpointError(f"checkpoint metadata is missing {key}.{sub}")


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".json")


def save_checkpoint(path: Path, model: KeywordCNN, metadata: Mapping[str, Any]) -> str:
    """Write ``path`` and its JSON sidecar atomically; return the ``.pt`` SHA-256."""
    meta = json.loads(json.dumps(dict(metadata)))  # must be JSON-serialisable
    _check_metadata(meta)
    state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    payload = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "metadata": meta,
        "model_state": state,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    digest = _sha256_file(path)
    sidecar = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_file": path.name,
        "checkpoint_sha256": digest,
        "state_dict": {
            name: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
            for name, t in state.items()
        },
        "metadata": meta,
    }
    side = sidecar_path(path)
    tmp_side = side.with_name(f".{side.name}.tmp")
    tmp_side.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_side.replace(side)
    return digest


def load_checkpoint(
    path: Path,
    *,
    expect_manifest_sha256: str | None = None,
    expect_labels: Sequence[str] | None = None,
    expect_window: WindowSection | None = None,
    expect_features: FeatureSection | None = None,
    require_sidecar: bool = True,
) -> LoadedCheckpoint:
    """Load and verify a checkpoint; return the model (eval mode) and its front end."""
    if not path.is_file():
        raise CheckpointError(f"checkpoint not found: {path}")
    digest = _sha256_file(path)
    if require_sidecar:
        side = sidecar_path(path)
        try:
            recorded = json.loads(side.read_text(encoding="utf-8"))["checkpoint_sha256"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise CheckpointError(f"checkpoint sidecar {side} is missing or unreadable") from exc
        if recorded != digest:
            raise CheckpointError(f"{path}: SHA-256 {digest} does not match sidecar {recorded}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # torch raises several unrelated types for bad files
        raise CheckpointError(f"{path}: not a loadable checkpoint: {exc}") from exc
    if not isinstance(payload, dict):
        raise CheckpointError(f"{path}: checkpoint payload must be a dict")
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(f"{path}: unknown checkpoint format {payload.get('format')!r}")
    if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise CheckpointError(
            f"{path}: checkpoint format version {payload.get('format_version')!r} is not supported"
        )
    metadata = payload.get("metadata")
    _check_metadata(metadata)
    assert isinstance(metadata, dict)
    state = payload.get("model_state")
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise CheckpointError(f"{path}: model_state must map names to tensors")

    model_meta = metadata["model"]
    if model_meta["architecture"] not in _SUPPORTED_ARCHITECTURES:
        raise CheckpointError(f"unsupported architecture {model_meta['architecture']!r}")
    prep = metadata["preprocessing"]
    if prep["feature_version"] != FEATURE_VERSION:
        raise CheckpointError(
            f"feature version {prep['feature_version']} != supported {FEATURE_VERSION}"
        )
    try:
        model_cfg = ModelSection.model_validate(model_meta["config"])
        window = WindowSection.model_validate(prep["window"])
        features = FeatureSection.model_validate(prep["features"])
    except ValidationError as exc:
        raise CheckpointError(f"checkpoint settings are invalid: {exc}") from exc
    fingerprint = feature_fingerprint(features, window)
    if fingerprint != prep["feature_fingerprint"]:
        raise CheckpointError(
            "feature fingerprint differs from this code's feature implementation "
            f"({prep['feature_fingerprint']} != {fingerprint})"
        )

    labels = metadata["labels"]
    expectations = {
        "manifest SHA-256": (expect_manifest_sha256, metadata["manifest"]["sha256"]),
        "labels": (list(expect_labels) if expect_labels is not None else None, labels),
        "window": (
            expect_window.model_dump(mode="json") if expect_window else None,
            prep["window"],
        ),
        "features": (
            expect_features.model_dump(mode="json") if expect_features else None,
            prep["features"],
        ),
    }
    for name, (expected, actual) in expectations.items():
        if expected is not None and expected != actual:
            raise CheckpointError(
                f"checkpoint {name} {actual!r} does not match expected {expected!r}"
            )

    if model_meta["num_classes"] != len(labels):
        raise CheckpointError("model num_classes does not match the number of labels")
    model = KeywordCNN(model_cfg, num_classes=len(labels), n_mels=features.n_mels)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"state dict does not match the architecture: {exc}") from exc
    if count_parameters(model) != model_meta["num_parameters"]:
        raise CheckpointError("parameter count does not match checkpoint metadata")
    model.eval()
    frontend = LogMelSpectrogram(features, window)
    return LoadedCheckpoint(model=model, frontend=frontend, metadata=metadata, sha256=digest)
