"""Tiny synthetic end-to-end training runs (no network, no real dataset)."""

from __future__ import annotations

import json
import runpy
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml

from robust_streaming_ml.checkpoint import load_checkpoint
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.training.data import DataCompatibilityError
from robust_streaming_ml.training.loop import TrainingError, train
from tests.conftest import git
from tests.training_fixtures import ToyTraining, make_toy_training


@pytest.fixture
def toy(tmp_path: Path, package_repo: Path) -> ToyTraining:
    return make_toy_training(tmp_path)


def log_records(run_path: Path) -> list[dict[str, Any]]:
    lines = (run_path / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_end_to_end_run_writes_all_artifacts(toy: ToyTraining, package_repo: Path) -> None:
    config = toy.config()
    result = train(config, dataset_spec=toy.spec)
    run = result.run.path
    files = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    assert files == {
        "config.yaml",
        "metadata.json",
        "train_log.jsonl",
        "summary.json",
        "checkpoints/best.pt",
        "checkpoints/best.json",
        "checkpoints/last.pt",
        "checkpoints/last.json",
    }
    records = log_records(run)
    assert [r["epoch"] for r in records] == [1, 2]
    assert all(r["train_examples"] == 14 and r["validation_examples"] == 2 for r in records)
    assert records[0]["learning_rate"] == 0.01 and records[1]["learning_rate"] == 0.005  # cosine

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary == result.summary
    assert summary["status"] == "completed"
    assert summary["rows_loaded"] == {"train": 14, "validation": 2, "test": 0}
    assert summary["manifest_sha256"] == toy.built.manifest_sha256
    best_epoch = min(records, key=lambda r: r["validation_loss"])["epoch"]
    assert summary["best"]["epoch"] == best_epoch

    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source"] == {"commit": git(package_repo, "rev-parse", "HEAD"), "dirty": False}
    assert (
        metadata["seeding"]["seed"] == 7 and metadata["seeding"]["torch_deterministic_algorithms"]
    )
    assert yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8")) == config.to_dict()

    loaded = load_checkpoint(
        result.best_checkpoint,
        expect_manifest_sha256=toy.built.manifest_sha256,
        expect_labels=["yes", "no"],
        expect_window=config.window,
        expect_features=config.features,
    )
    meta = loaded.metadata
    assert meta["epoch"] == best_epoch and meta["kind"] == "best"
    assert meta["run_id"] == result.run.run_id
    assert meta["source"] == metadata["source"]
    assert meta["seeding"]["applied"] == metadata["seeding"]
    assert meta["manifest"]["splits_used"] == ["train", "validation"]
    assert meta["config_sha256"] == metadata["config_sha256"]
    assert meta["model"]["num_parameters"] == summary["num_parameters"]
    assert loaded.frontend(torch.zeros(1, 16000)).shape == (1, 1, 16, 101)
    assert load_checkpoint(run / "checkpoints" / "last.pt").metadata["epoch"] == 2


def test_training_is_reproducible(toy: ToyTraining, tmp_path: Path) -> None:
    first = train(toy.config(), dataset_spec=toy.spec, root=tmp_path / "a")
    second = train(toy.config(), dataset_spec=toy.spec, root=tmp_path / "b")
    a = torch.load(first.run.path / "checkpoints" / "last.pt", weights_only=True)["model_state"]
    b = torch.load(second.run.path / "checkpoints" / "last.pt", weights_only=True)["model_state"]
    assert all(torch.equal(a[k], b[k]) for k in a)

    def strip(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in r.items() if k != "seconds"} for r in records]

    assert strip(log_records(first.run.path)) == strip(log_records(second.run.path))


def test_augmentation_changes_training_when_enabled(toy: ToyTraining, tmp_path: Path) -> None:
    plain = train(toy.config(), dataset_spec=toy.spec, root=tmp_path / "p")
    augmented = train(
        toy.config(**{"training.augmentation": {"enabled": True, "time_shift_max_ms": 50.0}}),
        dataset_spec=toy.spec,
        root=tmp_path / "q",
    )
    assert (
        log_records(plain.run.path)[0]["train_loss"]
        != log_records(augmented.run.path)[0]["train_loss"]
    )


def test_dirty_source_is_refused_before_anything_is_written(
    toy: ToyTraining, package_repo: Path, tmp_path: Path
) -> None:
    (package_repo / "uncommitted.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ProvenanceError, match="uncommitted"):
        train(toy.config(), dataset_spec=toy.spec)
    assert not (tmp_path / "runs").exists()
    result = train(toy.config(), dataset_spec=toy.spec, allow_dirty_source=True)
    assert result.run.metadata["source"]["dirty"] is True


def test_incompatible_manifest_is_refused_before_anything_is_written(
    toy: ToyTraining, tmp_path: Path
) -> None:
    with pytest.raises(DataCompatibilityError):
        train(toy.config(**{"data.manifest_sha256": "0" * 64}), dataset_spec=toy.spec)
    assert not (tmp_path / "runs").exists()


def test_failure_during_training_is_recorded(
    toy: ToyTraining, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def nan_loss(logits: torch.Tensor, labels: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return logits.sum() * float("nan")

    monkeypatch.setattr(torch.nn.functional, "cross_entropy", nan_loss)
    with pytest.raises(TrainingError, match="non-finite"):
        train(toy.config(), dataset_spec=toy.spec)
    (run_path,) = (tmp_path / "runs").iterdir()
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and "non-finite" in summary["error"]


def test_cli_rejects_an_invalid_config(tmp_path: Path) -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_baseline.py"
    main = runpy.run_path(str(script), run_name="train_baseline")["main"]
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\nseed: 1\n", encoding="utf-8")
    assert main(["--config", str(bad)]) == 1
