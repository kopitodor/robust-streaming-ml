"""Synthetic end-to-end evaluation (no network, no real dataset)."""

from __future__ import annotations

import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from robust_streaming_ml.evaluation import run as eval_run
from robust_streaming_ml.evaluation.calibration import load_calibration
from robust_streaming_ml.evaluation.config import EvaluationConfig
from robust_streaming_ml.evaluation.metrics import classification_metrics
from robust_streaming_ml.evaluation.predictions import decode_predictions
from robust_streaming_ml.evaluation.run import EvaluationError, evaluate
from robust_streaming_ml.run_dir import ProvenanceError
from robust_streaming_ml.training.data import DataCompatibilityError
from tests.conftest import git, git_commit_all
from tests.evaluation_fixtures import Trained, make_trained

EVAL_FILES = {
    "config.yaml",
    "metadata.json",
    "calibration.json",
    "report.json",
    "report.md",
    "summary.json",
    "predictions/validation.csv",
    "predictions/test.csv",
    "predictions/validation_calibrated.csv",
    "predictions/test_calibrated.csv",
}


@pytest.fixture
def trained(tmp_path: Path, package_repo: Path) -> Trained:
    return make_trained(tmp_path)


def eval_runs(trained: Trained) -> list[Path]:
    root = trained.tmp / "eval_runs"
    return sorted(root.iterdir()) if root.exists() else []


def rewrite_checkpoint(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    payload = torch.load(path, weights_only=True)
    mutate(payload["metadata"])
    torch.save(payload, path)
    side = path.with_suffix(".json")
    sidecar = json.loads(side.read_text(encoding="utf-8"))
    sidecar["checkpoint_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    side.write_text(json.dumps(sidecar), encoding="utf-8")


# --- end to end ------------------------------------------------------------------


def test_end_to_end_evaluation_writes_regenerable_artifacts(trained: Trained) -> None:
    result = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    run = result.run.path
    files = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    assert files == EVAL_FILES
    report = json.loads((run / "report.json").read_text(encoding="utf-8"))
    assert report == result.report
    assert report["protocol"]["id"] == "kw10-clean-eval-v1"
    assert report["manifest"]["rows_by_split"] == {"train": 14, "validation": 2, "test": 2}
    assert (
        report["checkpoint"]["sha256"]
        == hashlib.sha256(trained.checkpoint.read_bytes()).hexdigest()
    )
    for name, digest in report["artifacts_sha256"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == digest, name

    labels = ("yes", "no")
    temperature = report["calibration"]["temperature"]
    for split in ("validation", "test"):
        preds = decode_predictions(
            (run / "predictions" / f"{split}.csv").read_bytes(), split=split, labels=labels
        )
        assert len(preds) == 2
        raw = classification_metrics(preds.logits, preds.y_true, labels, n_bins=15)
        cal = classification_metrics(
            preds.logits, preds.y_true, labels, temperature=temperature, n_bins=15
        )
        cal["in_sample"] = split == "validation"
        assert report["metrics"][split] == {"raw": raw, "calibrated": cal}
        assert report["duplicate_sensitivity"][split]["rows_official"] == 2

    calibration = load_calibration(
        run / "calibration.json",
        expect_checkpoint_sha256=report["checkpoint"]["sha256"],
        expect_labels=labels,
    )
    assert calibration["fitted_on"]["split"] == "validation"
    assert calibration["fitted_on"]["examples"] == 2
    assert (
        calibration["fitted_on"]["predictions_sha256"]
        == report["artifacts_sha256"]["predictions/validation.csv"]
    )
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "completed" and summary["temperature"] == temperature
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    assert report["evaluation_source"] == metadata["source"]
    assert "Official results" in (run / "report.md").read_text(encoding="utf-8")


def test_evaluation_is_deterministic(trained: Trained) -> None:
    first = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    second = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    for name in ("predictions/validation.csv", "predictions/test.csv", "calibration.json"):
        assert (first.run.path / name).read_bytes() == (second.run.path / name).read_bytes()
    assert first.report["metrics"] == second.report["metrics"]


def test_calibration_is_persisted_before_any_test_inference(
    trained: Trained, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, bool]] = []
    original = eval_run._predict

    def spy(config: Any, loaded: Any, manifest: Any, window: Any, split: str) -> Any:
        (run_path,) = eval_runs(trained)
        calls.append((split, (run_path / "calibration.json").exists()))
        return original(config, loaded, manifest, window, split)

    monkeypatch.setattr(eval_run, "_predict", spy)
    evaluate(trained.config(), dataset_spec=trained.toy.spec)
    assert calls == [("validation", False), ("test", True)]


def test_train_split_is_never_evaluated(trained: Trained) -> None:
    config = trained.config()
    with pytest.raises(EvaluationError, match="not evaluated"):
        eval_run._predict(config, None, None, None, "train")  # type: ignore[arg-type]


def test_duplicate_sensitivity_keeps_one_row_per_audio_hash(trained: Trained) -> None:
    result = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    preds = decode_predictions(
        (result.run.path / "predictions" / "test.csv").read_bytes(),
        split="test",
        labels=("yes", "no"),
    )
    duplicate = preds.__class__(**{**preds.__dict__, "audio_sha256": (preds.audio_sha256[0],) * 2})
    assert eval_run._dedup_indices(duplicate) == [0]
    assert eval_run._dedup_indices(preds) == [0, 1]


# --- gates (nothing may be written) --------------------------------------------------


def assert_rejected(
    trained: Trained, config: EvaluationConfig, error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=match):
        evaluate(config, dataset_spec=trained.toy.spec)
    assert eval_runs(trained) == []


def test_pinned_checkpoint_hash_mismatch(trained: Trained) -> None:
    assert_rejected(
        trained, trained.config(**{"checkpoint.sha256": "0" * 64}), EvaluationError, "pinned"
    )


def test_missing_checkpoint(trained: Trained) -> None:
    config = trained.config(**{"checkpoint.path": str(trained.tmp / "missing.pt")})
    assert_rejected(trained, config, EvaluationError, "not found")


def test_checkpoint_for_another_manifest(trained: Trained) -> None:
    rewrite_checkpoint(trained.checkpoint, lambda m: m["manifest"].__setitem__("sha256", "0" * 64))
    assert_rejected(trained, trained.config(), EvaluationError, "manifest SHA-256")


def test_label_mismatch(trained: Trained) -> None:
    assert_rejected(
        trained, trained.config(**{"data.labels": ["no", "yes"]}), EvaluationError, "labels"
    )


def test_checkpoint_from_dirty_tree(trained: Trained) -> None:
    rewrite_checkpoint(trained.checkpoint, lambda m: m["source"].__setitem__("dirty", True))
    assert_rejected(trained, trained.config(), EvaluationError, "dirty working tree")


def test_checkpoint_commit_not_an_ancestor(trained: Trained, package_repo: Path) -> None:
    (package_repo / "side.txt").write_text("side\n", encoding="utf-8")
    git(package_repo, "add", "side.txt")
    git_commit_all(package_repo, "side branch commit")
    orphan = git(package_repo, "rev-parse", "HEAD")
    git(package_repo, "reset", "--hard", "--quiet", "HEAD~1")
    rewrite_checkpoint(trained.checkpoint, lambda m: m["source"].__setitem__("commit", orphan))
    assert_rejected(trained, trained.config(), EvaluationError, "not an ancestor")


def test_unknown_checkpoint_commit(trained: Trained) -> None:
    rewrite_checkpoint(trained.checkpoint, lambda m: m["source"].__setitem__("commit", "f" * 40))
    assert_rejected(trained, trained.config(), ProvenanceError, "cat-file")


def test_training_run_metadata_mismatch(trained: Trained) -> None:
    metadata_path = trained.checkpoint.parent.parent / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["config_sha256"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    assert_rejected(trained, trained.config(), EvaluationError, "config_sha256")
    metadata_path.unlink()
    assert_rejected(trained, trained.config(), EvaluationError, "metadata unreadable")


def test_unvalidated_manifest_is_refused(trained: Trained) -> None:
    report_path = trained.toy.built.report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["status"] = "failed"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert_rejected(trained, trained.config(), DataCompatibilityError, "status is 'failed'")


def test_dirty_evaluation_source_is_refused(trained: Trained, package_repo: Path) -> None:
    (package_repo / "wip.py").write_text("x = 1\n", encoding="utf-8")
    assert_rejected(trained, trained.config(), ProvenanceError, "uncommitted")


def test_failure_after_start_is_recorded(trained: Trained, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic failure")

    monkeypatch.setattr(eval_run, "fit_temperature", boom)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        evaluate(trained.config(), dataset_spec=trained.toy.spec)
    (run_path,) = eval_runs(trained)
    summary = json.loads((run_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and "synthetic failure" in summary["error"]
    assert not (run_path / "predictions" / "test.csv").exists()


def test_cli_rejects_a_mismatched_checkpoint(trained: Trained) -> None:
    path = trained.tmp / "eval.yaml"
    raw = trained.raw()
    raw["checkpoint"]["sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")  # JSON is valid YAML
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_baseline.py"
    main = runpy.run_path(str(script), run_name="evaluate_baseline")["main"]
    assert main(["--config", str(path)]) == 1
    assert eval_runs(trained) == []
