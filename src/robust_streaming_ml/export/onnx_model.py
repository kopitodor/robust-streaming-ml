"""ONNX export and loading for protocol ``kw10-onnx-v1`` (docs/streaming-onnx-protocol.md, §3).

The exported graph is the checkpoint's validated ``LogMelSpectrogram``
(``torch.stft`` becomes the ONNX ``STFT`` operator) followed by ``kws_cnn_v1``
in eval mode: ``waveform`` float32 ``[batch, W]`` -> ``logits`` float32
``[batch, C]``, opset 18, default domain only. Softmax and aggregation stay
outside the graph.

The model's ``metadata_props`` (keys ``rsml.<field>``, JSON values) tie it to
the checkpoint SHA-256, labels, window and feature settings with the feature
fingerprint, the manifest, the tensor contract, the export source revision,
and the export toolchain. A JSON sidecar ``<model>.json`` repeats them with
the file SHA-256, the export run, and a graph summary.

:func:`load_onnx_model` fails closed (:class:`OnnxModelError`) on any hash,
metadata, contract, opset, checker, or expectation mismatch.
:func:`create_session` builds an ONNX Runtime session that uses only the
``CPUExecutionProvider``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import torch
from pydantic import ValidationError
from torch import nn

from robust_streaming_ml.checkpoint import LoadedCheckpoint
from robust_streaming_ml.features import (
    FEATURE_VERSION,
    LogMelSpectrogram,
    feature_fingerprint,
)
from robust_streaming_ml.model import KeywordCNN
from robust_streaming_ml.streaming.windowing import STREAMING_PROTOCOL_ID
from robust_streaming_ml.training.config import FeatureSection, WindowSection

__all__ = [
    "EXPORT_TOOLCHAIN",
    "INPUT_NAME",
    "MAX_BATCH",
    "ONNX_FORMAT",
    "ONNX_FORMAT_VERSION",
    "ONNX_PROTOCOL_ID",
    "OPSET",
    "OUTPUT_NAME",
    "PROVIDER",
    "RUNTIME_VERSION",
    "LoadedOnnxModel",
    "OnnxModelError",
    "OnnxWindowBackend",
    "WaveformClassifier",
    "create_session",
    "export_onnx_bytes",
    "graph_summary",
    "installed_versions",
    "load_onnx_model",
    "model_metadata",
    "require_runtime",
    "require_toolchain",
    "session_options_record",
    "sidecar_path",
    "write_onnx_model",
]

ONNX_PROTOCOL_ID = "kw10-onnx-v1"
ONNX_FORMAT = "robust-streaming-ml/onnx"
ONNX_FORMAT_VERSION = 1
OPSET = 18
INPUT_NAME = "waveform"
OUTPUT_NAME = "logits"
MAX_BATCH = 65536
PROVIDER = "CPUExecutionProvider"
# Formal runs require exactly these versions (docs/streaming-onnx-protocol.md, §3).
EXPORT_TOOLCHAIN: dict[str, str] = {"torch": "2.14.0+cpu", "onnx": "1.23.0", "onnxscript": "0.7.2"}
RUNTIME_VERSION = "1.30.0"

_META_PREFIX = "rsml."
_META_KEYS = (
    "format",
    "format_version",
    "export_protocol",
    "streaming_protocol",
    "checkpoint",
    "labels",
    "preprocessing",
    "manifest",
    "model",
    "contract",
    "source",
    "toolchain",
)

Float32 = npt.NDArray[np.float32]


class OnnxModelError(RuntimeError):
    """An exported model, its toolchain, or its runtime is missing or incompatible."""


class WaveformClassifier(nn.Module):
    """Waveforms ``[batch, W]`` -> logits: the validated front end, then the CNN."""

    def __init__(self, frontend: LogMelSpectrogram, model: KeywordCNN) -> None:
        super().__init__()
        self.frontend = frontend
        self.model = model

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.model(self.frontend(waveform))
        return logits


def installed_versions() -> dict[str, str | None]:
    """Versions of the export toolchain and runtime that are importable here."""
    versions: dict[str, str | None] = {"torch": str(torch.__version__), "onnx": onnx.__version__}
    for name in ("onnxscript", "onnxruntime"):
        try:
            versions[name] = str(importlib.import_module(name).__version__)
        except ImportError:
            versions[name] = None
    return versions


def require_toolchain(expected: Mapping[str, str] | None = None) -> dict[str, str]:
    """The exact export toolchain of the protocol, or an error naming every difference."""
    wanted = dict(EXPORT_TOOLCHAIN if expected is None else expected)
    found = installed_versions()
    diffs = {name: (found.get(name), version) for name, version in wanted.items()}
    wrong = {name: pair for name, pair in diffs.items() if pair[0] != pair[1]}
    if wrong:
        detail = ", ".join(f"{n} {f!r} (protocol pins {e!r})" for n, (f, e) in wrong.items())
        raise OnnxModelError(f"export toolchain differs from {ONNX_PROTOCOL_ID}: {detail}")
    return {name: str(found[name]) for name in wanted}


def require_runtime(expected_version: str | None = None) -> dict[str, Any]:
    """ONNX Runtime at the pinned version with the CPU provider, or an error."""
    wanted = RUNTIME_VERSION if expected_version is None else expected_version
    try:
        import onnxruntime
    except ImportError as exc:
        raise OnnxModelError(f"onnxruntime is not installed (run 'uv sync'): {exc}") from exc
    version = str(onnxruntime.__version__)
    if version != wanted:
        raise OnnxModelError(
            f"onnxruntime {version} is installed, but {ONNX_PROTOCOL_ID} pins {wanted}; "
            "results would not be comparable"
        )
    available = list(onnxruntime.get_available_providers())
    if PROVIDER not in available:
        raise OnnxModelError(f"onnxruntime has no {PROVIDER} (available: {available})")
    return {"onnxruntime": version, "provider": PROVIDER, "available_providers": available}


def model_metadata(
    loaded: LoadedCheckpoint, *, source: Mapping[str, Any], toolchain: Mapping[str, str]
) -> dict[str, Any]:
    """The metadata stored in the model and its sidecar (no run IDs or times)."""
    meta = loaded.metadata
    prep = meta["preprocessing"]
    labels = list(meta["labels"])
    window = WindowSection.model_validate(prep["window"])
    return {
        "format": ONNX_FORMAT,
        "format_version": ONNX_FORMAT_VERSION,
        "export_protocol": ONNX_PROTOCOL_ID,
        "streaming_protocol": STREAMING_PROTOCOL_ID,
        "checkpoint": {
            "sha256": loaded.sha256,
            "run_id": meta["run_id"],
            "epoch": meta["epoch"],
            "kind": meta["kind"],
        },
        "labels": labels,
        "preprocessing": {
            "window": prep["window"],
            "features": prep["features"],
            "feature_version": prep["feature_version"],
            "feature_fingerprint": prep["feature_fingerprint"],
            "input_scaling": "float32(int16) / 32768",
        },
        "manifest": {"dataset": meta["manifest"]["dataset"], "sha256": meta["manifest"]["sha256"]},
        "model": {
            "architecture": meta["model"]["architecture"],
            "num_parameters": meta["model"]["num_parameters"],
        },
        "contract": {
            "opset": OPSET,
            "input": {
                "name": INPUT_NAME,
                "dtype": "float32",
                "shape": ["batch", window.num_samples],
            },
            "output": {"name": OUTPUT_NAME, "dtype": "float32", "shape": ["batch", len(labels)]},
            "max_batch": MAX_BATCH,
        },
        "source": {"commit": source["commit"], "dirty": source["dirty"]},
        "toolchain": dict(toolchain),
    }


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def export_onnx_bytes(loaded: LoadedCheckpoint, metadata: Mapping[str, Any]) -> bytes:
    """Export the checkpoint's front end + model and return the serialised ONNX model."""
    module = WaveformClassifier(loaded.frontend, loaded.model).eval()
    window_samples = int(loaded.frontend.window.num_samples)
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    program = torch.onnx.export(
        module,
        (torch.zeros(2, window_samples, dtype=torch.float32),),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        opset_version=OPSET,
        dynamo=True,
        verbose=False,
        report=False,
        dynamic_shapes=({0: batch},),
    )
    if program is None:
        raise OnnxModelError("torch.onnx.export returned no program")
    proto = program.model_proto
    del proto.metadata_props[:]
    for key in _META_KEYS:
        entry = proto.metadata_props.add()
        entry.key = _META_PREFIX + key
        entry.value = _canonical(metadata[key])
    data: bytes = proto.SerializeToString()
    _check_proto(onnx.load_model_from_string(data), dict(metadata))
    return data


def graph_summary(proto: onnx.ModelProto) -> dict[str, Any]:
    def shape(value: onnx.ValueInfoProto) -> list[str | int]:
        return [d.dim_param or d.dim_value for d in value.type.tensor_type.shape.dim]

    return {
        "inputs": [{"name": v.name, "shape": shape(v)} for v in proto.graph.input],
        "outputs": [{"name": v.name, "shape": shape(v)} for v in proto.graph.output],
        "opset_imports": [{"domain": o.domain, "version": o.version} for o in proto.opset_import],
        "ir_version": proto.ir_version,
        "producer": f"{proto.producer_name} {proto.producer_version}",
        "nodes": len(proto.graph.node),
        "op_types": sorted({n.op_type for n in proto.graph.node}),
        "initializers": len(proto.graph.initializer),
    }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + ".json")


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_onnx_model(
    path: Path,
    data: bytes,
    *,
    run_id: str,
    created_at_utc: str,
    environment: Mapping[str, Any],
) -> str:
    """Write the model and its sidecar atomically; return the model SHA-256."""
    proto = onnx.load_model_from_string(data)
    metadata = _decode_metadata(proto)
    digest = _sha256(data)
    sidecar = {
        "format": ONNX_FORMAT,
        "format_version": ONNX_FORMAT_VERSION,
        "onnx_file": path.name,
        "onnx_sha256": digest,
        "onnx_bytes": len(data),
        "export_run_id": run_id,
        "created_at_utc": created_at_utc,
        "graph": graph_summary(proto),
        "environment": dict(environment),
        "metadata": metadata,
    }
    _write_atomic(path, data)
    _write_atomic(
        sidecar_path(path), (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return digest


def _decode_metadata(proto: onnx.ModelProto) -> dict[str, Any]:
    props = {p.key: p.value for p in proto.metadata_props}
    missing = [k for k in _META_KEYS if _META_PREFIX + k not in props]
    if missing:
        raise OnnxModelError(f"model metadata is missing {missing}")
    try:
        metadata = {k: json.loads(props[_META_PREFIX + k]) for k in _META_KEYS}
    except ValueError as exc:
        raise OnnxModelError(f"model metadata is not valid JSON: {exc}") from exc
    if metadata["format"] != ONNX_FORMAT or metadata["format_version"] != ONNX_FORMAT_VERSION:
        raise OnnxModelError(
            f"unsupported model format {metadata['format']!r} "
            f"version {metadata['format_version']!r}"
        )
    if metadata["export_protocol"] != ONNX_PROTOCOL_ID:
        raise OnnxModelError(f"unsupported export protocol {metadata['export_protocol']!r}")
    return metadata


def _check_proto(proto: onnx.ModelProto, metadata: dict[str, Any]) -> None:
    """Opset, checker, and tensor contract against the metadata."""
    opsets = sorted((o.domain, o.version) for o in proto.opset_import)
    if opsets not in ([("", OPSET)], [("ai.onnx", OPSET)]):
        raise OnnxModelError(f"model opsets {opsets} != default domain opset {OPSET} only")
    try:
        onnx.checker.check_model(proto, full_check=True)
    except onnx.checker.ValidationError as exc:
        raise OnnxModelError(f"ONNX checker rejected the model: {exc}") from exc
    contract = metadata["contract"]
    if contract["opset"] != OPSET:
        raise OnnxModelError(f"metadata opset {contract['opset']} != {OPSET}")
    window_samples = int(metadata["preprocessing"]["window"]["num_samples"])
    num_labels = len(metadata["labels"])
    expected = {
        "input": (INPUT_NAME, window_samples, proto.graph.input),
        "output": (OUTPUT_NAME, num_labels, proto.graph.output),
    }
    for kind, (name, width, values) in expected.items():
        if len(values) != 1:
            raise OnnxModelError(f"model must have exactly one {kind}, has {len(values)}")
        value = values[0]
        tensor = value.type.tensor_type
        dims = list(tensor.shape.dim)
        ok = (
            value.name == name
            and tensor.elem_type == onnx.TensorProto.FLOAT
            and len(dims) == 2
            and bool(dims[0].dim_param)
            and dims[1].dim_value == width
        )
        if not ok or contract[kind] != {
            "name": name,
            "dtype": "float32",
            "shape": ["batch", width],
        }:
            raise OnnxModelError(
                f"model {kind} contract differs from {kind} {name!r} float32 [batch, {width}]"
            )


@dataclass(frozen=True)
class LoadedOnnxModel:
    path: Path
    data: bytes
    sha256: str
    metadata: dict[str, Any]
    sidecar: dict[str, Any]

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(self.metadata["labels"])

    @property
    def window(self) -> WindowSection:
        return WindowSection.model_validate(self.metadata["preprocessing"]["window"])


def load_onnx_model(
    path: Path,
    *,
    expect_checkpoint_sha256: str | None = None,
    expect_labels: Sequence[str] | None = None,
    expect_feature_fingerprint: str | None = None,
    expect_window: WindowSection | None = None,
    expect_features: FeatureSection | None = None,
    expect_manifest_sha256: str | None = None,
) -> LoadedOnnxModel:
    """Load and verify an exported model (see the module docstring)."""
    if not path.is_file():
        raise OnnxModelError(f"ONNX model not found: {path}")
    data = path.read_bytes()
    digest = _sha256(data)
    side = sidecar_path(path)
    try:
        sidecar = json.loads(side.read_text(encoding="utf-8"))
        recorded = sidecar["onnx_sha256"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise OnnxModelError(f"ONNX sidecar {side} is missing or unreadable") from exc
    if recorded != digest:
        raise OnnxModelError(f"{path}: SHA-256 {digest} does not match sidecar {recorded}")
    try:
        proto = onnx.load_model_from_string(data)
    except Exception as exc:  # protobuf raises DecodeError and others
        raise OnnxModelError(f"{path}: not a loadable ONNX model: {exc}") from exc
    metadata = _decode_metadata(proto)
    if sidecar.get("metadata") != metadata:
        raise OnnxModelError(f"{side}: sidecar metadata differs from the model's metadata")
    _check_proto(proto, metadata)

    prep = metadata["preprocessing"]
    if prep["feature_version"] != FEATURE_VERSION:
        raise OnnxModelError(f"feature version {prep['feature_version']} != {FEATURE_VERSION}")
    try:
        window = WindowSection.model_validate(prep["window"])
        features = FeatureSection.model_validate(prep["features"])
    except ValidationError as exc:
        raise OnnxModelError(f"model preprocessing settings are invalid: {exc}") from exc
    if feature_fingerprint(features, window) != prep["feature_fingerprint"]:
        raise OnnxModelError("feature fingerprint differs from this code's feature implementation")
    expectations = {
        "checkpoint SHA-256": (expect_checkpoint_sha256, metadata["checkpoint"]["sha256"]),
        "labels": (list(expect_labels) if expect_labels is not None else None, metadata["labels"]),
        "feature fingerprint": (expect_feature_fingerprint, prep["feature_fingerprint"]),
        "window": (
            expect_window.model_dump(mode="json") if expect_window else None,
            prep["window"],
        ),
        "features": (
            expect_features.model_dump(mode="json") if expect_features else None,
            prep["features"],
        ),
        "manifest SHA-256": (expect_manifest_sha256, metadata["manifest"]["sha256"]),
    }
    for name, (expected, actual) in expectations.items():
        if expected is not None and expected != actual:
            raise OnnxModelError(f"model {name} {actual!r} does not match expected {expected!r}")
    return LoadedOnnxModel(path=path, data=data, sha256=digest, metadata=metadata, sidecar=sidecar)


def session_options_record(*, intra_op_threads: int, inter_op_threads: int = 1) -> dict[str, Any]:
    return {
        "providers": [PROVIDER],
        "graph_optimization_level": "ORT_ENABLE_ALL",
        "execution_mode": "ORT_SEQUENTIAL",
        "intra_op_num_threads": intra_op_threads,
        "inter_op_num_threads": inter_op_threads,
    }


def create_session(
    model: LoadedOnnxModel, *, intra_op_threads: int = 1, inter_op_threads: int = 1
) -> Any:
    """An ONNX Runtime session on ``CPUExecutionProvider`` only, with a verified contract."""
    import onnxruntime as ort

    if intra_op_threads < 1 or inter_op_threads < 1:
        raise OnnxModelError("thread counts must be positive")
    if PROVIDER not in ort.get_available_providers():
        raise OnnxModelError(f"onnxruntime has no {PROVIDER}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.log_severity_level = 3
    session = ort.InferenceSession(model.data, options, providers=[PROVIDER])
    if list(session.get_providers()) != [PROVIDER]:
        raise OnnxModelError(f"session providers {session.get_providers()} != [{PROVIDER}]")
    width = model.window.num_samples
    inputs = [(i.name, i.type, i.shape[1:]) for i in session.get_inputs()]
    outputs = [(o.name, o.type, o.shape[1:]) for o in session.get_outputs()]
    if inputs != [(INPUT_NAME, "tensor(float)", [width])] or outputs != [
        (OUTPUT_NAME, "tensor(float)", [len(model.labels)])
    ]:
        raise OnnxModelError(f"session contract differs: inputs {inputs}, outputs {outputs}")
    return session


class OnnxWindowBackend:
    """The deployment path: an ONNX Runtime CPU session over float32 windows."""

    name = "onnxruntime"

    def __init__(self, session: Any) -> None:
        self.session = session

    def __call__(self, windows: Float32) -> Float32:
        batch = np.ascontiguousarray(windows, dtype=np.float32)
        (logits,) = self.session.run([OUTPUT_NAME], {INPUT_NAME: batch})
        return np.asarray(logits)
