"""A toy model with a Milestone 4-style baseline, and configs that export it (kw10-onnx-v1)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from robust_streaming_ml.config import parse_config
from robust_streaming_ml.evaluation.run import evaluate
from robust_streaming_ml.export import onnx_model
from robust_streaming_ml.export.config import ExportConfig
from robust_streaming_ml.export.onnx_model import installed_versions
from robust_streaming_ml.export.run import ExportResult, export_and_verify
from tests.audio_fixtures import tone, wav_bytes
from tests.evaluation_fixtures import Trained, apply_overrides, make_trained
from tests.training_fixtures import training_files


def files_with_long_test_clips() -> dict[str, bytes]:
    """Toy files whose two test clips (16,000 and 12,000 frames) form a 9-window stream."""
    files = training_files()
    files["yes/dddddddd_nohash_0.wav"] = wav_bytes(tone(16000, 71))
    files["no/dddddddd_nohash_0.wav"] = wav_bytes(tone(12000, 72))
    return files


def pin_installed_versions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the protocol versions to the installed ones (CI locks another onnxruntime)."""
    found = installed_versions()
    monkeypatch.setattr(
        onnx_model,
        "EXPORT_TOOLCHAIN",
        {k: str(found[k]) for k in ("torch", "onnx", "onnxscript")},
    )
    monkeypatch.setattr(onnx_model, "RUNTIME_VERSION", str(found["onnxruntime"]))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class ExportSetup:
    trained: Trained
    baseline_dir: Path

    @property
    def tmp(self) -> Path:
        return self.trained.tmp

    def raw(self) -> dict[str, Any]:
        evaluation = self.trained.raw()
        return {
            "schema_version": 1,
            "name": "toy-onnx",
            "seed": 7,
            "deterministic": True,
            "run": {"output_root": str(self.tmp / "export_runs")},
            "data": evaluation["data"],
            "checkpoint": evaluation["checkpoint"],
            "baseline": {
                "run_dir": str(self.baseline_dir),
                "report_sha256": sha256_file(self.baseline_dir / "report.json"),
            },
            "export": {"reference_batch_size": 3, "num_threads": 1},
        }

    def config(self, **overrides: Any) -> ExportConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=ExportConfig)

    def run(self, **overrides: Any) -> ExportResult:
        return export_and_verify(self.config(**overrides), dataset_spec=self.trained.toy.spec)

    def runs(self) -> list[Path]:
        root = self.tmp / "export_runs"
        return sorted(root.iterdir()) if root.exists() else []


def make_export_setup(tmp_path: Path) -> ExportSetup:
    """Train the toy model and evaluate it (requires ``package_repo`` for provenance)."""
    trained = make_trained(tmp_path, files_with_long_test_clips())
    baseline = evaluate(trained.config(), dataset_spec=trained.toy.spec)
    return ExportSetup(trained=trained, baseline_dir=baseline.run.path)
