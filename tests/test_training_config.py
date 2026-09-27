from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from robust_streaming_ml.config import ConfigError, load_config, parse_config
from robust_streaming_ml.data.speech_commands import SPEECH_COMMANDS_KW10
from robust_streaming_ml.training.config import TrainingConfig

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "kw10_cnn_baseline.yaml"


def baseline_dict() -> dict[str, Any]:
    data = yaml.safe_load(BASELINE.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_baseline_config_is_valid_and_matches_the_pinned_dataset() -> None:
    config = load_config(BASELINE, schema=TrainingConfig)
    assert isinstance(config, TrainingConfig)
    assert config.data.dataset == SPEECH_COMMANDS_KW10.manifest_name
    assert config.data.manifest_sha256 == SPEECH_COMMANDS_KW10.expected_manifest_sha256
    assert config.data.labels == list(SPEECH_COMMANDS_KW10.labels)
    assert config.training.augmentation.enabled is False
    assert (config.window.version, config.features.version, config.schema_version) == (1, 1, 1)


def test_plain_experiment_config_still_loads_as_before() -> None:
    example = BASELINE.with_name("example.yaml")
    assert type(load_config(example)).__name__ == "ExperimentConfig"


def test_unquoted_yaml_labels_become_booleans_and_are_rejected(tmp_path: Path) -> None:
    text = BASELINE.read_text(encoding="utf-8").replace(
        '["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]',
        "[yes, no, up, down, left, right, on, off, stop, go]",
    )
    path = tmp_path / "unquoted.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match="valid string"):
        load_config(path, schema=TrainingConfig)


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        (None, "schema_version", 2),
        (None, "unexpected", 1),
        ("data", "manifest_sha256", "abc"),
        ("data", "labels", ["yes", "yes"]),
        ("data", "labels", ["yes"]),
        ("training", "learning_rate", "1e-3"),  # YAML 1.1 reads 1e-3 as a string
        ("training", "epochs", 0),
        ("training", "optimizer", "sgd"),
        ("training", "lr_schedule", "step"),
        ("training", "num_workers", -1),
        ("window", "sample_rate_hz", 8000),  # f_max 8000 Hz would exceed Nyquist
        ("window", "num_samples", 300),  # shorter than win_length
        ("model", "architecture", "transformer"),
        ("checkpoint", "save_best", False),
    ],
)
def test_invalid_training_configs_are_rejected(section: str | None, key: str, value: Any) -> None:
    data = baseline_dict()
    (data if section is None else data[section])[key] = value
    with pytest.raises(ConfigError):
        parse_config(data, schema=TrainingConfig)


def test_augmentation_must_be_explicitly_enabled() -> None:
    data = baseline_dict()
    data["training"].pop("augmentation")
    config = parse_config(data, schema=TrainingConfig)
    assert config.training.augmentation.enabled is False
    data["training"]["augmentation"] = {"enabled": True, "time_shift_max_ms": 0.0}
    with pytest.raises(ConfigError):
        parse_config(data, schema=TrainingConfig)
