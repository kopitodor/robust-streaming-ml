"""A validated toy manifest plus a matching TrainingConfig, for training tests."""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_streaming_ml.config import parse_config
from robust_streaming_ml.data.validation import BuildResult, DatasetSpec, build_manifest
from robust_streaming_ml.training.config import TrainingConfig
from tests.audio_fixtures import TOY_LABELS, Toy, build_toy, default_files, tone, wav_bytes


def training_files() -> dict[str, bytes]:
    """Default toy files plus extra train utterances, so a batch has several examples."""
    files = default_files()
    seed = 500
    for label in TOY_LABELS:
        for index in range(2, 6):
            seed += 1
            files[f"{label}/aaaaaaaa_nohash_{index}.wav"] = wav_bytes(tone(1200 + seed, seed))
    return files


@dataclass
class ToyTraining:
    toy: Toy
    built: BuildResult
    spec: DatasetSpec
    raw: dict[str, Any]

    def config(self, **overrides: Any) -> TrainingConfig:
        """Config from ``raw`` with dotted-path overrides, e.g. ``{"training.epochs": 3}``."""
        data = copy.deepcopy(self.raw)
        for dotted, value in overrides.items():
            target = data
            *parents, leaf = dotted.split(".")
            for key in parents:
                target = target.setdefault(key, {})
            target[leaf] = value
        return parse_config(data, schema=TrainingConfig)


def make_toy_training(tmp_path: Path, files: dict[str, bytes] | None = None) -> ToyTraining:
    """Build and validate a toy dataset (requires the ``package_repo`` fixture)."""
    toy = build_toy(tmp_path, files=training_files() if files is None else files)
    built = build_manifest(
        toy.spec,
        toy.root,
        toy.archive,
        manifest_dir=tmp_path / "manifests",
        report_dir=tmp_path / "reports",
    )
    spec = dataclasses.replace(toy.spec, expected_manifest_sha256=built.manifest_sha256)
    raw: dict[str, Any] = {
        "schema_version": 1,
        "name": "toy-train",
        "seed": 7,
        "deterministic": True,
        "run": {"output_root": str(tmp_path / "runs")},
        "data": {
            "dataset": spec.manifest_name,
            "manifest_path": str(built.manifest_path),
            "manifest_sha256": built.manifest_sha256,
            "validation_report_path": str(built.report_path),
            "audio_root": str(toy.root),
            "labels": list(TOY_LABELS),
            "load_workers": 2,
        },
        "features": {"n_mels": 16},
        "model": {"channels": [4, 8], "dropout": 0.1},
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "validation_batch_size": 3,
            "learning_rate": 0.01,
            "num_threads": 1,
        },
    }
    return ToyTraining(toy=toy, built=built, spec=spec, raw=raw)
