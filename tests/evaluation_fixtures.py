"""A toy model trained on the synthetic dataset, and configs that evaluate it."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_streaming_ml.config import parse_config
from robust_streaming_ml.evaluation.config import EvaluationConfig
from robust_streaming_ml.training.loop import train
from tests.training_fixtures import ToyTraining, make_toy_training


def apply_overrides(data: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Dotted-path overrides, e.g. ``{"robustness.batch_size": 4}``."""
    data = copy.deepcopy(data)
    for dotted, value in overrides.items():
        target = data
        *parents, leaf = dotted.split(".")
        for key in parents:
            target = target[key]
        target[leaf] = value
    return data


@dataclass
class Trained:
    toy: ToyTraining
    checkpoint: Path
    tmp: Path

    def raw(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "name": "toy-eval",
            "seed": 7,
            "deterministic": True,
            "run": {"output_root": str(self.tmp / "eval_runs")},
            "data": dict(self.toy.raw["data"]),
            "checkpoint": {
                "path": str(self.checkpoint),
                "sha256": hashlib.sha256(self.checkpoint.read_bytes()).hexdigest(),
            },
            "evaluation": {"batch_size": 3, "num_threads": 1},
        }

    def config(self, **overrides: Any) -> EvaluationConfig:
        return parse_config(apply_overrides(self.raw(), overrides), schema=EvaluationConfig)


def make_trained(tmp_path: Path) -> Trained:
    """Train the toy model (requires the ``package_repo`` fixture for provenance)."""
    toy = make_toy_training(tmp_path)
    result = train(toy.config(), dataset_spec=toy.spec)
    return Trained(toy=toy, checkpoint=result.best_checkpoint, tmp=tmp_path)
