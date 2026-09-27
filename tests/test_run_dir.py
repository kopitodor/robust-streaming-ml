from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pytest
import yaml

from robust_streaming_ml.config import ExperimentConfig, load_config, parse_config
from robust_streaming_ml.run_dir import (
    CONFIG_FILENAME,
    METADATA_FILENAME,
    collect_environment_metadata,
    config_digest,
    create_run_dir,
)


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return parse_config(
        {"name": "unit", "seed": 17, "run": {"output_root": str(tmp_path / "runs")}}
    )


def test_creates_directory_under_configured_root(config: ExperimentConfig) -> None:
    run = create_run_dir(config)
    assert run.path.is_dir()
    assert run.path.parent == config.run.output_root
    assert run.path.name == run.run_id
    assert sorted(p.name for p in run.path.iterdir()) == [CONFIG_FILENAME, METADATA_FILENAME]


def test_default_run_id_format(config: ExperimentConfig) -> None:
    run = create_run_dir(config)
    assert re.fullmatch(r"\d{8}T\d{6}Z_unit_[0-9a-f]{8}", run.run_id)


def test_default_run_ids_are_unique(config: ExperimentConfig) -> None:
    ids = {create_run_dir(config).run_id for _ in range(5)}
    assert len(ids) == 5


def test_explicit_root_and_run_id(config: ExperimentConfig, tmp_path: Path) -> None:
    run = create_run_dir(config, root=tmp_path / "elsewhere", run_id="fixed-id")
    assert run.path == tmp_path / "elsewhere" / "fixed-id"


def test_existing_run_directory_is_never_reused(config: ExperimentConfig) -> None:
    create_run_dir(config, run_id="dup")
    with pytest.raises(FileExistsError):
        create_run_dir(config, run_id="dup")


@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "has space"])
def test_unsafe_run_id_rejected(config: ExperimentConfig, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        create_run_dir(config, run_id=run_id)


def test_recorded_config_round_trips(config: ExperimentConfig) -> None:
    run = create_run_dir(config)
    assert load_config(run.config_path) == config
    assert yaml.safe_load(run.config_path.read_text(encoding="utf-8")) == config.to_dict()


def test_metadata_file_contents(config: ExperimentConfig) -> None:
    run = create_run_dir(config)
    recorded = json.loads(run.metadata_path.read_text(encoding="utf-8"))
    assert recorded == run.metadata
    assert recorded["schema_version"] == 1
    assert recorded["run_id"] == run.run_id
    assert recorded["experiment_name"] == "unit"
    assert recorded["seed"] == 17
    assert recorded["deterministic"] is True
    assert recorded["config_file"] == CONFIG_FILENAME
    assert recorded["config_sha256"] == config_digest(config)
    assert datetime.fromisoformat(recorded["created_at_utc"]).utcoffset() is not None


def test_environment_metadata_fields() -> None:
    env = collect_environment_metadata()
    assert set(env) == {"python", "platform", "packages", "torch", "git", "argv"}
    assert env["python"]["version"]
    assert env["platform"]["system"]
    for name in ("robust-streaming-ml", "numpy", "pydantic", "pyyaml", "torch"):
        assert env["packages"][name], name
    assert isinstance(env["torch"]["cuda_available"], bool)
    assert isinstance(env["torch"]["num_threads"], int)
    json.dumps(env)  # must be serializable


def test_git_metadata_absent_outside_repository(tmp_path: Path) -> None:
    assert collect_environment_metadata(cwd=tmp_path)["git"] is None


def test_config_digest_is_stable_and_sensitive() -> None:
    a = parse_config({"name": "x", "seed": 1})
    assert config_digest(a) == config_digest(parse_config({"seed": 1, "name": "x"}))
    assert config_digest(a) != config_digest(parse_config({"name": "x", "seed": 2}))
    assert re.fullmatch(r"[0-9a-f]{64}", config_digest(a))
