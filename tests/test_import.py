"""Import smoke tests.

Also run by ``scripts/check_dist.py`` against the installed wheel in an
isolated environment, so keep this file self-contained (no conftest or
repository-relative imports).
"""

from __future__ import annotations

import importlib
from importlib import metadata, resources

import robust_streaming_ml


def test_package_version() -> None:
    assert robust_streaming_ml.__version__ == "0.1.0"


def test_distribution_metadata_matches_package() -> None:
    assert metadata.version("robust-streaming-ml") == robust_streaming_ml.__version__


def test_package_is_marked_typed() -> None:
    assert resources.files("robust_streaming_ml").joinpath("py.typed").is_file()


def test_public_modules_import() -> None:
    modules = ("config", "seed", "run_dir", "data.acquisition", "data.manifest", "data.wav")
    training = ("features", "model", "checkpoint", "training.config", "training.data")
    evaluation = ("evaluation.metrics", "evaluation.predictions", "evaluation.calibration")
    extra = (
        "evaluation.config",
        "evaluation.run",
        "training.loop",
        "robustness.corruptions",
        "robustness.config",
        "robustness.run",
    )
    for name in (
        *modules,
        "data.validation",
        "data.speech_commands",
        *training,
        *evaluation,
        *extra,
    ):
        importlib.import_module(f"robust_streaming_ml.{name}")
