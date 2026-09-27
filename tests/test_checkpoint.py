from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from robust_streaming_ml.checkpoint import (
    CHECKPOINT_FORMAT,
    CHECKPOINT_FORMAT_VERSION,
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
    sidecar_path,
)
from robust_streaming_ml.features import FEATURE_VERSION, feature_fingerprint
from robust_streaming_ml.model import KeywordCNN, count_parameters
from robust_streaming_ml.training.config import FeatureSection, ModelSection, WindowSection

LABELS = ["yes", "no", "up"]
MANIFEST_SHA = "f" * 64
FEATURES = FeatureSection(n_mels=16)
WINDOW = WindowSection()
MODEL_CFG = ModelSection(channels=[4, 8])


def make_model() -> KeywordCNN:
    torch.manual_seed(0)
    return KeywordCNN(MODEL_CFG, num_classes=len(LABELS), n_mels=FEATURES.n_mels)


def metadata(model: KeywordCNN) -> dict[str, Any]:
    return {
        "kind": "best",
        "epoch": 3,
        "run_id": "run-1",
        "labels": LABELS,
        "config": {"name": "unit"},
        "config_sha256": "c" * 64,
        "model": {
            "architecture": "kws_cnn_v1",
            "config": MODEL_CFG.model_dump(mode="json"),
            "num_classes": len(LABELS),
            "num_parameters": count_parameters(model),
            "input_shape": [1, 16, 101],
        },
        "preprocessing": {
            "window": WINDOW.model_dump(mode="json"),
            "features": FEATURES.model_dump(mode="json"),
            "feature_version": FEATURE_VERSION,
            "feature_fingerprint": feature_fingerprint(FEATURES, WINDOW),
        },
        "manifest": {"dataset": "toy", "sha256": MANIFEST_SHA, "rows_by_split": {"train": 5}},
        "source": {"commit": "a" * 40, "dirty": False},
        "seeding": {"applied": {"seed": 7}},
        "validation": {"loss": 0.5, "accuracy": 0.75, "epoch": 3},
        "torch_version": torch.__version__,
    }


@pytest.fixture
def saved(tmp_path: Path) -> tuple[Path, KeywordCNN]:
    model = make_model()
    path = tmp_path / "ckpt" / "best.pt"
    save_checkpoint(path, model, metadata(model))
    return path, model


def rewrite(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Rewrite the payload with ``mutate`` applied and refresh the sidecar hash."""
    payload = torch.load(path, weights_only=True)
    mutate(payload)
    torch.save(payload, path)
    side = sidecar_path(path)
    data = json.loads(side.read_text(encoding="utf-8"))
    data["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    side.write_text(json.dumps(data), encoding="utf-8")


def test_round_trip_restores_weights_and_frontend(saved: tuple[Path, KeywordCNN]) -> None:
    path, model = saved
    loaded = load_checkpoint(
        path,
        expect_manifest_sha256=MANIFEST_SHA,
        expect_labels=LABELS,
        expect_window=WINDOW,
        expect_features=FEATURES,
    )
    original = model.state_dict()
    restored = loaded.model.state_dict()
    assert set(original) == set(restored)
    assert all(torch.equal(original[k], restored[k]) for k in original)
    assert not loaded.model.training
    model.eval()
    x = torch.randn(2, 16000) * 0.1
    assert torch.equal(loaded.model(loaded.frontend(x)), model(loaded.frontend(x)))
    assert loaded.metadata["validation"] == {"loss": 0.5, "accuracy": 0.75, "epoch": 3}
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_sidecar_is_human_readable_and_complete(saved: tuple[Path, KeywordCNN]) -> None:
    path, model = saved
    side = json.loads(sidecar_path(path).read_text(encoding="utf-8"))
    assert side["format"] == CHECKPOINT_FORMAT
    assert side["format_version"] == CHECKPOINT_FORMAT_VERSION
    assert side["checkpoint_file"] == "best.pt"
    assert side["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert side["metadata"]["manifest"]["sha256"] == MANIFEST_SHA
    assert side["state_dict"]["classifier.weight"] == {"shape": [3, 8], "dtype": "float32"}
    assert set(side["state_dict"]) == set(model.state_dict())


def test_payload_is_plain_data_loadable_with_weights_only(saved: tuple[Path, KeywordCNN]) -> None:
    payload = torch.load(saved[0], weights_only=True)
    assert set(payload) == {"format", "format_version", "metadata", "model_state"}


def test_save_rejects_incomplete_or_unserialisable_metadata(tmp_path: Path) -> None:
    model = make_model()
    incomplete = metadata(model)
    del incomplete["manifest"]
    with pytest.raises(CheckpointError, match="manifest"):
        save_checkpoint(tmp_path / "a.pt", model, incomplete)
    with pytest.raises(TypeError):
        save_checkpoint(tmp_path / "b.pt", model, {**metadata(model), "bad": object()})
    assert not (tmp_path / "a.pt").exists()


def test_missing_file_and_sidecar(saved: tuple[Path, KeywordCNN], tmp_path: Path) -> None:
    with pytest.raises(CheckpointError, match="not found"):
        load_checkpoint(tmp_path / "nope.pt")
    path, _ = saved
    sidecar_path(path).unlink()
    with pytest.raises(CheckpointError, match="sidecar"):
        load_checkpoint(path)
    load_checkpoint(path, require_sidecar=False)  # explicit opt-out still verifies the rest


def test_modified_file_fails_the_sidecar_hash(saved: tuple[Path, KeywordCNN]) -> None:
    path, _ = saved
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(CheckpointError, match="does not match sidecar"):
        load_checkpoint(path)


def test_garbage_and_arbitrary_pickles_are_rejected(tmp_path: Path) -> None:
    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"not a checkpoint")
    with pytest.raises(CheckpointError, match="not a loadable"):
        load_checkpoint(garbage, require_sidecar=False)
    unsafe = tmp_path / "unsafe.pt"
    torch.save({"format": CHECKPOINT_FORMAT, "obj": Path("x")}, unsafe)
    with pytest.raises(CheckpointError, match="not a loadable"):
        load_checkpoint(unsafe, require_sidecar=False)


def _set(keys: tuple[str, ...], value: Any) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        target = payload
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value

    return mutate


def _delete(keys: tuple[str, ...]) -> Callable[[dict[str, Any]], None]:
    def mutate(payload: dict[str, Any]) -> None:
        target = payload
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_set(("format",), "other"), "unknown checkpoint format"),
        (_set(("format_version",), 2), "format version"),
        (_delete(("metadata", "source")), "missing 'source'"),
        (_delete(("metadata", "preprocessing", "feature_fingerprint")), "feature_fingerprint"),
        (_set(("metadata", "model", "architecture"), "resnet"), "unsupported architecture"),
        (_set(("metadata", "preprocessing", "feature_version"), 99), "feature version"),
        (_set(("metadata", "preprocessing", "feature_fingerprint"), "0" * 64), "fingerprint"),
        (_set(("metadata", "preprocessing", "features", "n_mels"), 20), "fingerprint"),
        (_set(("metadata", "preprocessing", "features", "hop_length"), 0), "settings are invalid"),
        (_set(("metadata", "model", "config", "channels"), [4, 16]), "state dict"),
        (_set(("metadata", "model", "num_classes"), 4), "num_classes"),
        (_set(("metadata", "labels"), ["yes", "no"]), "num_classes"),
        (_set(("metadata", "model", "num_parameters"), 1), "parameter count"),
        (_set(("model_state",), {"x": 1}), "model_state"),
    ],
)
def test_incompatible_checkpoints_are_rejected(
    saved: tuple[Path, KeywordCNN], mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    path, _ = saved
    rewrite(path, mutate)
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(path)


def test_unexpected_extra_weights_are_rejected(saved: tuple[Path, KeywordCNN]) -> None:
    path, _ = saved

    def add_weight(payload: dict[str, Any]) -> None:
        payload["model_state"] = {**payload["model_state"], "extra.weight": torch.zeros(1)}

    rewrite(path, add_weight)
    with pytest.raises(CheckpointError, match="state dict"):
        load_checkpoint(path)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"expect_manifest_sha256": "0" * 64}, "manifest SHA-256"),
        ({"expect_labels": ["no", "yes", "up"]}, "labels"),
        ({"expect_window": WindowSection(num_samples=8000)}, "window"),
        ({"expect_features": FeatureSection(n_mels=32)}, "features"),
    ],
)
def test_expectation_mismatches_are_rejected(
    saved: tuple[Path, KeywordCNN], kwargs: dict[str, Any], message: str
) -> None:
    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(saved[0], **copy.deepcopy(kwargs))
