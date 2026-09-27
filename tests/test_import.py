from __future__ import annotations

import importlib

import robust_streaming_ml


def test_package_version() -> None:
    assert robust_streaming_ml.__version__ == "0.1.0"


def test_public_modules_import() -> None:
    for name in ("config", "seed", "run_dir"):
        importlib.import_module(f"robust_streaming_ml.{name}")
