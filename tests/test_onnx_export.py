"""ONNX export, loading, runtime, and parity (kw10-onnx-v1) on a small random-weight model."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime
import pytest
import torch

from robust_streaming_ml.checkpoint import LoadedCheckpoint
from robust_streaming_ml.export import onnx_model
from robust_streaming_ml.export.onnx_model import (
    INPUT_NAME,
    OPSET,
    OUTPUT_NAME,
    LoadedOnnxModel,
    OnnxModelError,
    OnnxWindowBackend,
    create_session,
    export_onnx_bytes,
    installed_versions,
    load_onnx_model,
    model_metadata,
    require_runtime,
    require_toolchain,
    sidecar_path,
    write_onnx_model,
)
from robust_streaming_ml.export.parity import (
    LOGIT_ATOL,
    PROBABILITY_ATOL,
    causal_aggregates,
    compare_aggregates,
    compare_logits,
)
from robust_streaming_ml.features import FEATURE_VERSION, LogMelSpectrogram, feature_fingerprint
from robust_streaming_ml.model import KeywordCNN, count_parameters
from robust_streaming_ml.streaming.classifier import (
    TorchWindowBackend,
    classify_offline,
    classify_stream,
)
from robust_streaming_ml.streaming.windowing import StreamingSpec
from robust_streaming_ml.training.config import FeatureSection, ModelSection, WindowSection

LABELS = ["yes", "no", "up"]
CHECKPOINT_SHA = "ab" * 32
MANIFEST_SHA = "cd" * 32


def toy_checkpoint() -> LoadedCheckpoint:
    torch.manual_seed(1234)
    window, features = WindowSection(), FeatureSection(n_mels=16)
    model_cfg = ModelSection(channels=[4, 8])
    model = KeywordCNN(model_cfg, num_classes=len(LABELS), n_mels=16)
    for module in model.modules():  # non-trivial batch-norm statistics
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.running_mean is not None and module.running_var is not None
            module.running_mean.uniform_(-1.0, 1.0)
            module.running_var.uniform_(0.5, 2.0)
    model.eval()
    metadata = {
        "kind": "best",
        "epoch": 3,
        "run_id": "toy-run",
        "labels": LABELS,
        "preprocessing": {
            "window": window.model_dump(mode="json"),
            "features": features.model_dump(mode="json"),
            "feature_version": FEATURE_VERSION,
            "feature_fingerprint": feature_fingerprint(features, window),
        },
        "manifest": {"dataset": "toy_v1_kw2", "sha256": MANIFEST_SHA},
        "model": {"architecture": "kws_cnn_v1", "num_parameters": count_parameters(model)},
    }
    frontend = LogMelSpectrogram(features, window)
    return LoadedCheckpoint(
        model=model, frontend=frontend, metadata=metadata, sha256=CHECKPOINT_SHA
    )


@dataclass
class Exported:
    loaded: LoadedCheckpoint
    metadata: dict[str, Any]
    data: bytes

    def write(self, directory: Path, data: bytes | None = None) -> Path:
        path = directory / "model.onnx"
        write_onnx_model(
            path,
            self.data if data is None else data,
            run_id="r",
            created_at_utc="t",
            environment={},
        )
        return path


@pytest.fixture(scope="module")
def exported() -> Exported:
    loaded = toy_checkpoint()
    metadata = model_metadata(
        loaded,
        source={"commit": "1" * 40, "dirty": False},
        toolchain={"torch": "t", "onnx": "o", "onnxscript": "s"},
    )
    return Exported(loaded=loaded, metadata=metadata, data=export_onnx_bytes(loaded, metadata))


def fixtures() -> npt.NDArray[np.float32]:
    """Fixed synthetic parity fixtures: tones, noise at several levels, silence, full scale."""
    rng = np.random.default_rng(20260928)
    t = np.arange(16000) / 16000
    rows = [
        0.5 * np.sin(2 * np.pi * 440 * t),
        0.05 * np.sin(2 * np.pi * 3000 * t) * (t < 0.4),
        rng.standard_normal(16000) * 0.1,
        rng.standard_normal(16000) * 0.001,
        np.zeros(16000),
        np.clip(rng.standard_normal(16000), -1.0, 32767 / 32768),
    ]
    quantised = [np.round(np.asarray(r) * 32768) / 32768 for r in rows]
    return np.stack(quantised).astype(np.float32)


def rewrite(exported: Exported, directory: Path, edit: Callable[[onnx.ModelProto], None]) -> Path:
    """Write a modified model with a consistent (re-hashed) sidecar."""
    proto = onnx.load_model_from_string(exported.data)
    edit(proto)
    path = directory / "model.onnx"
    data = proto.SerializeToString()
    directory.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    side = json.loads(sidecar_path(exported.write(directory / "orig")).read_text("utf-8"))
    side["onnx_sha256"] = __import__("hashlib").sha256(data).hexdigest()
    props = {p.key[len("rsml.") :]: json.loads(p.value) for p in proto.metadata_props}
    side["metadata"] = props
    sidecar_path(path).write_text(json.dumps(side), encoding="utf-8")
    return path


def set_prop(key: str, value: Any) -> Callable[[onnx.ModelProto], None]:
    def edit(proto: onnx.ModelProto) -> None:
        for prop in proto.metadata_props:
            if prop.key == "rsml." + key:
                prop.value = json.dumps(value)

    return edit


# --- export ------------------------------------------------------------------------------


def test_export_is_byte_deterministic(exported: Exported) -> None:
    assert export_onnx_bytes(exported.loaded, exported.metadata) == exported.data


def test_exported_graph_matches_the_contract(exported: Exported) -> None:
    proto = onnx.load_model_from_string(exported.data)
    assert [(o.domain, o.version) for o in proto.opset_import] == [("", OPSET)]
    assert OPSET == 18
    (inp,), (out,) = proto.graph.input, proto.graph.output
    dims = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    assert inp.name == INPUT_NAME and dims[1] == 16000 and isinstance(dims[0], str)
    assert out.name == OUTPUT_NAME and out.type.tensor_type.shape.dim[1].dim_value == 3
    ops = {n.op_type for n in proto.graph.node}
    assert "STFT" in ops and "Softmax" not in ops and "Dropout" not in ops
    props = {p.key: json.loads(p.value) for p in proto.metadata_props}
    assert props["rsml.checkpoint"]["sha256"] == CHECKPOINT_SHA
    assert props["rsml.labels"] == LABELS
    assert props["rsml.contract"]["input"] == {
        "name": "waveform",
        "dtype": "float32",
        "shape": ["batch", 16000],
    }
    assert props["rsml.source"] == {"commit": "1" * 40, "dirty": False}
    # no export run ID or time in the model bytes: only the fixed metadata fields
    assert sorted(props) == sorted(f"rsml.{k}" for k in exported.metadata)


def test_round_trip_with_every_expectation(exported: Exported, tmp_path: Path) -> None:
    path = exported.write(tmp_path)
    prep = exported.metadata["preprocessing"]
    model = load_onnx_model(
        path,
        expect_checkpoint_sha256=CHECKPOINT_SHA,
        expect_labels=LABELS,
        expect_feature_fingerprint=prep["feature_fingerprint"],
        expect_window=WindowSection(),
        expect_features=FeatureSection(n_mels=16),
        expect_manifest_sha256=MANIFEST_SHA,
    )
    assert model.labels == tuple(LABELS) and model.window == WindowSection()
    assert model.metadata == exported.metadata
    sidecar = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
    assert sidecar["onnx_sha256"] == model.sha256 and sidecar["graph"]["op_types"]


# --- runtime and parity fixtures ---------------------------------------------------------


@pytest.fixture
def session_model(exported: Exported, tmp_path: Path) -> tuple[LoadedOnnxModel, Any]:
    model = load_onnx_model(exported.write(tmp_path))
    return model, create_session(model)


def test_onnx_runtime_matches_pytorch_on_fixed_fixtures(
    exported: Exported, session_model: tuple[LoadedOnnxModel, Any]
) -> None:
    _, session = session_model
    windows = fixtures()
    with torch.no_grad():
        reference = exported.loaded.model(exported.loaded.frontend(torch.from_numpy(windows)))
    backend = OnnxWindowBackend(session)
    per_window = np.concatenate([backend(windows[i : i + 1]) for i in range(len(windows))])
    batched = backend(windows)
    result = compare_logits(reference.numpy(), per_window)
    assert result["passed"], result
    assert result["abs_logit_diff"]["max"] < LOGIT_ATOL / 10
    assert per_window.dtype == np.float32 and batched.shape == (6, 3)
    np.testing.assert_allclose(batched, per_window, atol=1e-4)


def test_onnx_streaming_equals_onnx_offline(session_model: tuple[LoadedOnnxModel, Any]) -> None:
    model, session = session_model
    spec = StreamingSpec.from_window(model.window)
    backend = OnnxWindowBackend(session)
    signal = (np.random.default_rng(9).standard_normal(21000) * 5000).astype(np.int16)
    reference, _ = classify_offline(signal, backend, spec, model.labels)
    got, decision = classify_stream(
        np.split(signal, [3, 7000, 16500]), backend, spec, model.labels, sample_rate_hz=16000
    )
    assert len(got) == len(reference) == 5
    for a, b in zip(got, reference, strict=True):
        assert np.array_equal(a.logits, b.logits)
        assert np.array_equal(a.aggregate_probabilities, b.aggregate_probabilities)
    assert decision is not None and decision.windows == 5


def test_onnx_and_torch_streams_agree_within_tolerance(
    exported: Exported, session_model: tuple[LoadedOnnxModel, Any]
) -> None:
    model, session = session_model
    spec = StreamingSpec.from_window(model.window)
    signal = (np.random.default_rng(4).standard_normal(30000) * 2000).astype(np.int16)
    torch_preds, _ = classify_offline(
        signal,
        TorchWindowBackend(exported.loaded.model, exported.loaded.frontend),
        spec,
        model.labels,
    )
    onnx_preds, _ = classify_offline(signal, OnnxWindowBackend(session), spec, model.labels)
    logits = compare_logits(
        np.stack([p.logits for p in torch_preds]), np.stack([p.logits for p in onnx_preds])
    )
    aggregates = compare_aggregates(
        np.stack([p.aggregate_probabilities for p in torch_preds]),
        np.stack([p.aggregate_probabilities for p in onnx_preds]),
    )
    assert logits["passed"] and aggregates["passed"] and logits["windows"] == 10


# --- loading fails closed ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("expectation", "match"),
    [
        ({"expect_checkpoint_sha256": "ef" * 32}, "checkpoint SHA-256"),
        ({"expect_labels": ["no", "yes", "up"]}, "labels"),
        ({"expect_feature_fingerprint": "0" * 64}, "feature fingerprint"),
        ({"expect_window": WindowSection(num_samples=16001)}, "window"),
        ({"expect_features": FeatureSection(n_mels=40)}, "features"),
        ({"expect_manifest_sha256": "0" * 64}, "manifest SHA-256"),
    ],
)
def test_expectation_mismatches_are_rejected(
    exported: Exported, tmp_path: Path, expectation: dict[str, Any], match: str
) -> None:
    path = exported.write(tmp_path)
    with pytest.raises(OnnxModelError, match=match):
        load_onnx_model(path, **expectation)


def test_missing_or_mismatched_sidecar_is_rejected(exported: Exported, tmp_path: Path) -> None:
    path = exported.write(tmp_path)
    side = sidecar_path(path)
    record = json.loads(side.read_text(encoding="utf-8"))
    record["onnx_sha256"] = "0" * 64
    side.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(OnnxModelError, match="does not match sidecar"):
        load_onnx_model(path)
    side.unlink()
    with pytest.raises(OnnxModelError, match="missing or unreadable"):
        load_onnx_model(path)
    with pytest.raises(OnnxModelError, match="not found"):
        load_onnx_model(tmp_path / "absent.onnx")


def test_sidecar_metadata_must_equal_the_model_metadata(exported: Exported, tmp_path: Path) -> None:
    path = exported.write(tmp_path)
    side = sidecar_path(path)
    record = json.loads(side.read_text(encoding="utf-8"))
    record["metadata"]["labels"] = ["no", "yes", "up"]
    side.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(OnnxModelError, match="sidecar metadata differs"):
        load_onnx_model(path)


def test_changed_model_metadata_is_caught_by_expectations(
    exported: Exported, tmp_path: Path
) -> None:
    path = rewrite(exported, tmp_path, set_prop("checkpoint", {"sha256": "ef" * 32}))
    loaded = load_onnx_model(path)  # consistent on its own ...
    assert loaded.metadata["checkpoint"]["sha256"] == "ef" * 32
    with pytest.raises(OnnxModelError, match="checkpoint SHA-256"):  # ... but not the pinned one
        load_onnx_model(path, expect_checkpoint_sha256=CHECKPOINT_SHA)


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (set_prop("format_version", 2), "unsupported model format"),
        (set_prop("export_protocol", "kw10-onnx-v0"), "unsupported export protocol"),
        (set_prop("labels", ["yes", "no"]), "output contract"),
        (
            set_prop(
                "preprocessing",
                {
                    **toy_checkpoint().metadata["preprocessing"],
                    "feature_fingerprint": "0" * 64,
                },
            ),
            "differs from this code",
        ),
    ],
)
def test_invalid_metadata_is_rejected(
    exported: Exported, tmp_path: Path, edit: Callable[[onnx.ModelProto], None], match: str
) -> None:
    with pytest.raises(OnnxModelError, match=match):
        load_onnx_model(rewrite(exported, tmp_path, edit))


def test_missing_metadata_is_rejected(exported: Exported, tmp_path: Path) -> None:
    def drop(proto: onnx.ModelProto) -> None:
        keep = [p for p in proto.metadata_props if p.key != "rsml.labels"]
        del proto.metadata_props[:]
        proto.metadata_props.extend(keep)

    with pytest.raises(OnnxModelError, match="missing"):
        load_onnx_model(rewrite(exported, tmp_path, drop))


def test_other_opsets_and_tensor_contracts_are_rejected(exported: Exported, tmp_path: Path) -> None:
    def opset(proto: onnx.ModelProto) -> None:
        proto.opset_import[0].version = 17

    with pytest.raises(OnnxModelError, match="opset"):
        load_onnx_model(rewrite(exported, tmp_path / "a", opset))

    def rename(proto: onnx.ModelProto) -> None:
        proto.graph.output[0].name = "scores"
        for node in proto.graph.node:
            node.output[:] = ["scores" if o == OUTPUT_NAME else o for o in node.output]

    with pytest.raises(OnnxModelError, match="output contract"):
        load_onnx_model(rewrite(exported, tmp_path / "b", rename))


def test_corrupt_model_bytes_are_rejected(exported: Exported, tmp_path: Path) -> None:
    path = exported.write(tmp_path, data=exported.data)
    garbage = b"\x08\x07not an onnx model"
    path.write_bytes(garbage)
    side = sidecar_path(path)
    record = json.loads(side.read_text(encoding="utf-8"))
    record["onnx_sha256"] = __import__("hashlib").sha256(garbage).hexdigest()
    side.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(OnnxModelError):
        load_onnx_model(path)


# --- runtime and toolchain gates ---------------------------------------------------------


def test_session_uses_only_the_cpu_provider(session_model: tuple[LoadedOnnxModel, Any]) -> None:
    _, session = session_model
    assert session.get_providers() == ["CPUExecutionProvider"]


def test_missing_cpu_provider_fails(
    exported: Exported, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = load_onnx_model(exported.write(tmp_path))
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["OtherProvider"])
    with pytest.raises(OnnxModelError, match="CPUExecutionProvider"):
        create_session(model)
    with pytest.raises(OnnxModelError, match="CPUExecutionProvider"):
        require_runtime(str(onnxruntime.__version__))


def test_runtime_version_is_pinned() -> None:
    installed = str(onnxruntime.__version__)
    assert require_runtime(installed)["onnxruntime"] == installed
    with pytest.raises(OnnxModelError, match=r"pins 0\.0\.1"):
        require_runtime("0.0.1")
    assert onnx_model.RUNTIME_VERSION == "1.30.0"


def test_export_toolchain_is_pinned() -> None:
    found = installed_versions()
    exact = {k: str(found[k]) for k in ("torch", "onnx", "onnxscript")}
    assert require_toolchain(exact) == exact
    with pytest.raises(OnnxModelError, match=r"onnx '.*' \(protocol pins '0\.0'\)"):
        require_toolchain({**exact, "onnx": "0.0"})
    assert onnx_model.EXPORT_TOOLCHAIN == {
        "torch": "2.14.0+cpu",
        "onnx": "1.23.0",
        "onnxscript": "0.7.2",
    }


def test_bad_thread_counts_are_rejected(session_model: tuple[LoadedOnnxModel, Any]) -> None:
    model, _ = session_model
    with pytest.raises(OnnxModelError, match="positive"):
        create_session(model, intra_op_threads=0)


# --- parity arithmetic (fixed fixtures) ----------------------------------------------------


def logits(*rows: list[float]) -> npt.NDArray[np.float32]:
    return np.asarray(rows, dtype=np.float32)


def test_identical_outputs_pass() -> None:
    ref = logits([1.0, 2.0, 3.0], [0.5, -1.0, 0.0])
    result = compare_logits(ref, ref.copy())
    assert result["passed"] and result["abs_logit_diff"]["max"] == 0.0
    assert result["top1_disagreements"] == 0 and result["violations"] == []


def test_logit_tolerance_violation() -> None:
    ref = logits([5.0, 0.0, -5.0])
    result = compare_logits(ref, ref + np.float32(2 * LOGIT_ATOL), row_ids=["w0"])
    # a uniform shift leaves the probabilities unchanged: only the logit bound fails
    assert not result["passed"] and result["violations_by_kind"]["logit"] == 1
    assert result["violations_by_kind"]["probability"] == 0
    assert result["violations"][0]["id"] == "w0" and result["violating_windows"] == 1


def test_probability_tolerance_violation_within_the_logit_tolerance() -> None:
    ref = logits([0.0, 0.0, -20.0])
    cand = logits([0.009, -0.009, -20.0])  # |dlogit| < 1e-2, |dprob| about 4.5e-3
    result = compare_logits(ref, cand)
    assert result["violations_by_kind"] == {"logit": 0, "probability": 1, "top1": 0}
    assert result["abs_probability_diff"]["max"] > PROBABILITY_ATOL
    assert result["top1_near_tie_flips"] == 0  # the reference tie resolves to index 0 in both


def test_top1_flip_outside_the_near_tie_margin_is_a_violation() -> None:
    result = compare_logits(logits([1.0, 0.9, 0.0]), logits([0.9, 1.0, 0.0]))
    assert result["violations_by_kind"]["top1"] == 1 and not result["passed"]


def test_near_tie_flip_is_counted_not_failed() -> None:
    result = compare_logits(logits([1.0, 0.999, -3.0]), logits([0.9995, 1.0, -3.0]))
    assert result["passed"] and result["top1_disagreements"] == 1
    assert result["top1_near_tie_flips"] == 1


def test_parity_rejects_mismatched_or_non_finite_inputs() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        compare_logits(logits([1.0, 2.0]), logits([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="non-finite"):
        compare_logits(logits([1.0, np.nan]), logits([1.0, 2.0]))


def test_aggregate_parity_and_regeneration() -> None:
    probs = np.asarray([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4], [0.5, 0.5]], dtype=np.float64)
    agg = causal_aggregates(probs, 2)
    np.testing.assert_allclose(agg[1], [0.45, 0.55])
    np.testing.assert_allclose(agg[3], [0.55, 0.45])
    assert compare_aggregates(agg, agg.copy())["passed"]
    shifted = agg.copy()
    shifted[2] += [0.002, -0.002]
    result = compare_aggregates(agg, shifted, row_ids=["a", "b", "c", "d"])
    assert not result["passed"] and result["violations"][0]["id"] == "c"
    near = agg.copy()
    near[3] = [0.4995, 0.5005]  # reference margin 0.1 -> a flip here is a violation
    assert not compare_aggregates(agg, near)["passed"]


def test_causal_aggregates_equal_the_streaming_aggregates(exported: Exported) -> None:
    spec = StreamingSpec.from_window(WindowSection())
    signal = (np.random.default_rng(2).standard_normal(26000) * 3000).astype(np.int16)
    backend = TorchWindowBackend(exported.loaded.model, exported.loaded.frontend)
    predictions, _ = classify_offline(signal, backend, spec, LABELS)
    probs = np.stack([p.probabilities for p in predictions])
    regenerated = causal_aggregates(probs, spec.aggregation_windows)
    assert np.array_equal(regenerated, np.stack([p.aggregate_probabilities for p in predictions]))
