from __future__ import annotations

import inspect
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

import robust_streaming_ml
from robust_streaming_ml import run_dir
from robust_streaming_ml.config import ExperimentConfig, load_config, parse_config
from robust_streaming_ml.run_dir import (
    CONFIG_FILENAME,
    METADATA_FILENAME,
    ProvenanceError,
    SourceRevision,
    collect_environment_metadata,
    config_digest,
    resolve_source_revision,
    start_run,
)
from robust_streaming_ml.seed import SeedState, seed_everything
from tests.conftest import git, git_commit_all

ORIGINAL_PACKAGE_DIR = run_dir.PACKAGE_DIR  # captured before any monkeypatching


@pytest.fixture
def config(tmp_path: Path) -> ExperimentConfig:
    return parse_config(
        {"name": "unit", "seed": 17, "run": {"output_root": str(tmp_path / "runs")}}
    )


@pytest.fixture(autouse=True)
def package_dir(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Stand in a temporary repository for the package's source directory."""
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", git_repo)
    return git_repo


def read_metadata(path: Path) -> dict[str, Any]:
    data = json.loads((path / METADATA_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


# --- directory layout -------------------------------------------------------


def test_creates_directory_under_configured_root(config: ExperimentConfig, git_repo: Path) -> None:
    run = start_run(config)
    assert run.path.is_dir()
    assert run.path.parent == config.run.output_root
    assert run.path.name == run.run_id
    assert sorted(p.name for p in run.path.iterdir()) == [CONFIG_FILENAME, METADATA_FILENAME]


def test_default_run_id_format(config: ExperimentConfig, git_repo: Path) -> None:
    run = start_run(config)
    assert re.fullmatch(r"\d{8}T\d{6}Z_unit_[0-9a-f]{8}", run.run_id)


def test_default_run_ids_are_unique(config: ExperimentConfig, git_repo: Path) -> None:
    ids = {start_run(config).run_id for _ in range(5)}
    assert len(ids) == 5


def test_explicit_root_and_run_id(config: ExperimentConfig, git_repo: Path, tmp_path: Path) -> None:
    run = start_run(config, root=tmp_path / "elsewhere", run_id="fixed-id")
    assert run.path == tmp_path / "elsewhere" / "fixed-id"


def test_existing_run_directory_is_never_reused(config: ExperimentConfig, git_repo: Path) -> None:
    start_run(config, run_id="dup")
    with pytest.raises(FileExistsError):
        start_run(config, run_id="dup")


@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "has space"])
def test_unsafe_run_id_rejected(config: ExperimentConfig, git_repo: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run_id"):
        start_run(config, run_id=run_id)


def test_recorded_config_round_trips(config: ExperimentConfig, git_repo: Path) -> None:
    run = start_run(config)
    assert load_config(run.config_path) == config
    assert yaml.safe_load(run.config_path.read_text(encoding="utf-8")) == config.to_dict()


def test_metadata_file_contents(config: ExperimentConfig, git_repo: Path) -> None:
    run = start_run(config)
    recorded = read_metadata(run.path)
    assert recorded == run.metadata
    assert recorded["schema_version"] == 2
    assert recorded["run_id"] == run.run_id
    assert recorded["experiment_name"] == "unit"
    assert recorded["config_file"] == CONFIG_FILENAME
    assert recorded["config_sha256"] == config_digest(config)
    assert datetime.fromisoformat(recorded["created_at_utc"]).utcoffset() is not None


# --- source provenance ------------------------------------------------------


def test_records_head_commit_of_clean_repository(config: ExperimentConfig, git_repo: Path) -> None:
    run = start_run(config)
    assert read_metadata(run.path)["source"] == {
        "commit": git(git_repo, "rev-parse", "HEAD"),
        "dirty": False,
    }


def test_records_new_commit_after_commit(config: ExperimentConfig, git_repo: Path) -> None:
    first = start_run(config).metadata["source"]["commit"]
    (git_repo / "fixture.txt").write_text("v2\n", encoding="utf-8")
    git_commit_all(git_repo, "v2")
    second = start_run(config).metadata["source"]
    assert second == {"commit": git(git_repo, "rev-parse", "HEAD"), "dirty": False}
    assert second["commit"] != first


def test_resolves_from_tracked_repository_subdirectory(git_repo: Path) -> None:
    sub = git_repo / "pkg"
    sub.mkdir()
    (sub / "mod.py").write_text("x = 1\n", encoding="utf-8")
    git(git_repo, "add", "pkg/mod.py")
    git_commit_all(git_repo, "add pkg")
    assert resolve_source_revision(sub) == SourceRevision(
        commit=git(git_repo, "rev-parse", "HEAD"), dirty=False
    )


# --- provenance source is fixed to the package location --------------------


def test_start_run_does_not_accept_a_source_directory(
    config: ExperimentConfig, tmp_path: Path
) -> None:
    assert "source_dir" not in inspect.signature(start_run).parameters
    with pytest.raises(TypeError):
        start_run(config, source_dir=tmp_path)  # type: ignore[call-arg]
    assert not config.run.output_root.exists()


def test_start_run_records_package_repository_not_working_directory(
    config: ExperimentConfig, git_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = tmp_path / "unrelated"
    other.mkdir()
    git(other, "init", "--quiet")
    (other / "other.txt").write_text("other\n", encoding="utf-8")
    git(other, "add", "other.txt")
    git_commit_all(other, "unrelated")
    monkeypatch.chdir(other)
    run = start_run(config)
    assert run.metadata["source"]["commit"] == git(git_repo, "rev-parse", "HEAD")
    assert run.metadata["source"]["commit"] != git(other, "rev-parse", "HEAD")


def test_untracked_package_directory_inside_repository_fails(
    config: ExperimentConfig, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy installed into an ignored venv inside a checkout has no revision."""
    (git_repo / ".git" / "info" / "exclude").write_text(".venv/\n", encoding="utf-8")
    installed = git_repo / ".venv" / "site-packages" / "robust_streaming_ml"
    installed.mkdir(parents=True)
    (installed / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", installed)
    with pytest.raises(ProvenanceError, match="no files tracked"):
        start_run(config)
    assert not config.run.output_root.exists()


def test_default_package_dir_is_the_imported_package() -> None:
    assert Path(robust_streaming_ml.__file__).resolve().parent == ORIGINAL_PACKAGE_DIR


def test_modified_tracked_file_marks_dirty(config: ExperimentConfig, git_repo: Path) -> None:
    (git_repo / "fixture.txt").write_text("changed\n", encoding="utf-8")
    run = start_run(config)
    assert read_metadata(run.path)["source"] == {
        "commit": git(git_repo, "rev-parse", "HEAD"),
        "dirty": True,
    }


def test_untracked_file_marks_dirty(git_repo: Path) -> None:
    (git_repo / "new.py").write_text("x = 1\n", encoding="utf-8")
    assert resolve_source_revision(git_repo).dirty is True


def test_ignored_file_does_not_mark_dirty(git_repo: Path) -> None:
    (git_repo / ".git" / "info" / "exclude").write_text("runs/\n", encoding="utf-8")
    (git_repo / "runs").mkdir()
    (git_repo / "runs" / "output.json").write_text("{}\n", encoding="utf-8")
    assert resolve_source_revision(git_repo) == SourceRevision(
        commit=git(git_repo, "rev-parse", "HEAD"), dirty=False
    )


def test_outside_repository_fails_with_git_diagnostic(
    config: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    not_repo = tmp_path / "plain"
    not_repo.mkdir()
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", not_repo)
    with pytest.raises(ProvenanceError, match=r"(?is)cannot resolve source revision.*git"):
        start_run(config)
    assert not config.run.output_root.exists()


def test_repository_without_commits_fails(
    config: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "--quiet")
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", empty)
    with pytest.raises(ProvenanceError, match="rev-parse"):
        start_run(config)
    assert not config.run.output_root.exists()


def test_missing_git_executable_fails(
    config: ExperimentConfig, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_dir, "_GIT_EXECUTABLE", "git-executable-that-does-not-exist")
    with pytest.raises(ProvenanceError, match="cannot run git"):
        start_run(config)
    assert not config.run.output_root.exists()


def test_provenance_failure_happens_before_seeding(
    config: ExperimentConfig, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(run_dir, "PACKAGE_DIR", tmp_path)
    seed_everything(999, deterministic=False)
    with pytest.raises(ProvenanceError):
        start_run(config)
    assert not torch.are_deterministic_algorithms_enabled()


# --- applied seeding --------------------------------------------------------


def test_start_run_applies_seed(config: ExperimentConfig, git_repo: Path) -> None:
    seed_everything(999, deterministic=False)  # deliberately different prior state
    start_run(config)
    after_run = (random.random(), torch.rand(3))
    seed_everything(config.seed, deterministic=config.deterministic)
    expected = (random.random(), torch.rand(3))
    assert after_run[0] == expected[0]
    assert torch.equal(after_run[1], expected[1])


@pytest.mark.parametrize("deterministic", [True, False])
def test_metadata_records_observed_seed_state(
    tmp_path: Path, git_repo: Path, deterministic: bool
) -> None:
    config = parse_config(
        {
            "name": "unit",
            "seed": 17,
            "deterministic": deterministic,
            "run": {"output_root": str(tmp_path / "runs")},
        }
    )
    seed_everything(999, deterministic=not deterministic)
    run = start_run(config)
    seeding = read_metadata(run.path)["seeding"]
    assert torch.are_deterministic_algorithms_enabled() is deterministic
    assert seeding["seed"] == 17
    assert seeding["deterministic"] is deterministic
    assert seeding["torch_deterministic_algorithms"] is deterministic
    assert seeding["cudnn_deterministic"] is deterministic
    assert seeding["cudnn_benchmark"] is (not deterministic)
    assert seeding["PYTHONHASHSEED"] == "17"


def test_seeding_that_did_not_take_effect_is_rejected(
    config: ExperimentConfig, git_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_seed_everything(seed: int, *, deterministic: bool = True) -> SeedState:
        return SeedState(seed=seed, deterministic=deterministic)  # applies nothing

    monkeypatch.setattr(run_dir, "seed_everything", fake_seed_everything)
    torch.use_deterministic_algorithms(False)
    with pytest.raises(RuntimeError, match="seeding did not take effect"):
        start_run(config)
    assert not config.run.output_root.exists()


# --- environment and digest -------------------------------------------------


def test_environment_metadata_fields() -> None:
    env = collect_environment_metadata()
    assert set(env) == {"python", "platform", "packages", "torch", "argv"}
    assert env["python"]["version"]
    assert env["platform"]["system"]
    for name in ("robust-streaming-ml", "numpy", "pydantic", "pyyaml", "torch"):
        assert env["packages"][name], name
    assert isinstance(env["torch"]["cuda_available"], bool)
    assert isinstance(env["torch"]["num_threads"], int)
    json.dumps(env)  # must be serializable


def test_config_digest_is_stable_and_sensitive() -> None:
    a = parse_config({"name": "x", "seed": 1})
    assert config_digest(a) == config_digest(parse_config({"seed": 1, "name": "x"}))
    assert config_digest(a) != config_digest(parse_config({"name": "x", "seed": 2}))
    assert re.fullmatch(r"[0-9a-f]{64}", config_digest(a))
