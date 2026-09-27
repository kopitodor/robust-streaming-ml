from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch


@pytest.fixture(autouse=True)
def restore_global_state(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Seeding mutates process-global state; undo it after each test."""
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def git(repo: Path, *args: str) -> str:
    """Run git in ``repo`` and return its stripped stdout."""
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()


def git_commit_all(repo: Path, message: str) -> None:
    """Commit all changes to tracked and staged files, independent of user git config."""
    git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "--all",
        "-m",
        message,
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """A temporary git repository with one committed fixture file."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "--quiet")
    (repo / "fixture.txt").write_text("fixture\n", encoding="utf-8")
    git(repo, "add", "fixture.txt")
    git_commit_all(repo, "fixture")
    return repo


@pytest.fixture
def package_repo(git_repo: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Make the temporary repository stand in for the package source (for provenance)."""
    from robust_streaming_ml import run_dir

    monkeypatch.setattr(run_dir, "PACKAGE_DIR", git_repo)
    return git_repo
