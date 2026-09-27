from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from robust_streaming_ml.config import (
    MAX_SEED,
    ConfigError,
    ExperimentConfig,
    load_config,
    parse_config,
)

VALID: dict[str, Any] = {"name": "baseline-v1", "seed": 1234}


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_valid_config_applies_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, "name: baseline-v1\nseed: 1234\n"))
    assert config == ExperimentConfig(name="baseline-v1", seed=1234)
    assert config.deterministic is True
    assert config.run.output_root == Path("runs")


def test_load_full_config(tmp_path: Path) -> None:
    text = "name: exp.a_1\nseed: 0\ndeterministic: false\nrun:\n  output_root: out/runs\n"
    config = load_config(write(tmp_path, text))
    assert config.seed == 0
    assert config.deterministic is False
    assert config.run.output_root == Path("out/runs")


def test_repository_example_config_is_valid() -> None:
    example = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    assert load_config(example).name == "example"


def test_to_dict_round_trips() -> None:
    config = parse_config({**VALID, "run": {"output_root": "x/y"}})
    assert parse_config(config.to_dict()) == config


def test_config_is_immutable() -> None:
    config = parse_config(VALID)
    with pytest.raises(ValueError, match="frozen"):
        config.seed = 5  # type: ignore[misc]


@pytest.mark.parametrize("seed", [0, MAX_SEED])
def test_seed_bounds_accepted(seed: int) -> None:
    assert parse_config({**VALID, "seed": seed}).seed == seed


@pytest.mark.parametrize(
    ("override", "field"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": MAX_SEED + 1}, "seed"),
        ({"seed": "42"}, "seed"),
        ({"seed": 4.0}, "seed"),
        ({"seed": True}, "seed"),
        ({"deterministic": "yes"}, "deterministic"),
        ({"name": ""}, "name"),
        ({"name": "has space"}, "name"),
        ({"name": "../escape"}, "name"),
        ({"name": "x" * 65}, "name"),
        ({"unknown_key": 1}, "unknown_key"),
        ({"run": {"output_root": "runs", "extra": 1}}, "extra"),
    ],
)
def test_invalid_values_rejected(override: dict[str, Any], field: str) -> None:
    with pytest.raises(ConfigError, match=field):
        parse_config({**VALID, **override})


@pytest.mark.parametrize("missing", ["name", "seed"])
def test_required_fields(missing: str) -> None:
    data = {k: v for k, v in VALID.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        parse_config(data)


@pytest.mark.parametrize("text", ["", "- a\n- b\n", "42\n"])
def test_non_mapping_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ConfigError, match="must be a mapping"):
        load_config(write(tmp_path, text))


def test_malformed_yaml_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(tmp_path, "name: [unclosed\n"))


def test_missing_file_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "absent.yaml")


def test_error_names_source_file(tmp_path: Path) -> None:
    path = write(tmp_path, "name: ok\nseed: -3\n")
    with pytest.raises(ConfigError, match=r"config\.yaml"):
        load_config(path)
