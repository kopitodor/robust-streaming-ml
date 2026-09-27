from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from robust_streaming_ml.evaluation.calibration import (
    CalibrationError,
    calibration_record,
    fit_temperature,
    load_calibration,
    save_calibration,
)
from robust_streaming_ml.evaluation.metrics import log_softmax, negative_log_likelihood
from robust_streaming_ml.evaluation.predictions import PredictionSet

LABELS = ("a", "b", "c", "d")


def synthetic(split: str, *, scale: float, n: int = 4000, seed: int = 0) -> PredictionSet:
    """Labels drawn from softmax(z); logits = scale * z, so the best T is about ``scale``."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 2.0, size=(n, len(LABELS)))
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(len(LABELS), p=row) for row in p], dtype=np.int64)
    return PredictionSet(
        split=split,
        labels=LABELS,
        relative_paths=tuple(f"a/{i:08x}_nohash_0.wav" for i in range(n)),
        audio_sha256=tuple(f"{i:064x}" for i in range(n)),
        speaker_ids=tuple(f"{i:08x}" for i in range(n)),
        y_true=y,
        logits=(scale * z).astype(np.float32),
    )


@pytest.mark.parametrize("split", ["test", "train"])
def test_fitting_on_anything_but_validation_is_refused(split: str) -> None:
    with pytest.raises(CalibrationError, match="only be fitted on 'validation'"):
        fit_temperature(synthetic(split, scale=1.0, n=10))


def test_recovers_a_known_temperature() -> None:
    fit = fit_temperature(synthetic("validation", scale=3.0))
    assert fit.temperature == pytest.approx(3.0, rel=0.1)
    assert fit.nll_after < fit.nll_before and not fit.at_bound
    assert fit.temperature * fit.inverse_temperature == pytest.approx(1.0)
    assert fit.examples == 4000


def test_matches_a_brute_force_grid_minimum() -> None:
    data = synthetic("validation", scale=0.5, n=1500, seed=3)
    fit = fit_temperature(data)
    grid = np.linspace(0.05, 20.0, 40001)
    nll = [negative_log_likelihood(log_softmax(data.logits, 1 / b), data.y_true) for b in grid]
    best = grid[int(np.argmin(nll))]
    assert fit.inverse_temperature == pytest.approx(best, abs=2 * (grid[1] - grid[0]))
    assert fit.nll_after <= min(nll) + 1e-12


def test_fit_is_deterministic_and_scale_equivariant() -> None:
    data = synthetic("validation", scale=1.5, n=1000, seed=5)
    assert fit_temperature(data) == fit_temperature(data)
    doubled = PredictionSet(**{**data.__dict__, "logits": (data.logits * 2).astype(np.float32)})
    assert fit_temperature(doubled).temperature == pytest.approx(
        2 * fit_temperature(data).temperature, rel=1e-6
    )


def test_optimum_outside_the_search_range_is_flagged() -> None:
    data = synthetic("validation", scale=1.0, n=500, seed=1)
    flat = PredictionSet(**{**data.__dict__, "logits": (data.logits * 1e-4).astype(np.float32)})
    assert fit_temperature(flat).at_bound is True


def record_for(fit_split: str = "validation") -> dict[str, Any]:
    fit = fit_temperature(synthetic("validation", scale=2.0, n=300))
    record = calibration_record(
        fit,
        fitted_on={"examples": 300, "predictions_sha256": "e" * 64},
        compatibility={"checkpoint_sha256": "c" * 64, "labels": list(LABELS)},
    )
    record["fitted_on"]["split"] = fit_split
    return record


def test_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "calibration.json"
    record = record_for()
    save_calibration(path, record)
    loaded = load_calibration(path, expect_checkpoint_sha256="c" * 64, expect_labels=LABELS)
    assert loaded == record
    assert loaded["parameters"]["temperature"] == record["parameters"]["temperature"]
    assert loaded["search"]["bounds"] == [0.05, 20.0] and loaded["search"]["iterations"] == 200


def test_saving_a_non_validation_fit_is_refused(tmp_path: Path) -> None:
    with pytest.raises(CalibrationError, match="not fitted on validation"):
        save_calibration(tmp_path / "c.json", record_for("test"))


@pytest.mark.parametrize(
    ("mutate", "kwargs", "message"),
    [
        (lambda r: r["fitted_on"].__setitem__("split", "test"), {}, "not fitted on 'validation'"),
        (lambda r: r.__setitem__("format_version", 2), {}, "unsupported calibration format"),
        (lambda r: r.__setitem__("method", "isotonic"), {}, "unsupported method"),
        (lambda r: r["parameters"].__setitem__("temperature", -1.0), {}, "invalid temperature"),
        (lambda r: r["parameters"].__setitem__("inverse_temperature", 9.0), {}, "disagree"),
        (lambda r: None, {"expect_checkpoint_sha256": "0" * 64}, "another checkpoint"),
        (lambda r: None, {"expect_labels": ["a", "b"]}, "labels"),
    ],
)
def test_incompatible_calibration_files_are_rejected(
    tmp_path: Path, mutate: Any, kwargs: dict[str, Any], message: str
) -> None:
    record = record_for()
    mutate(record)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    expectations: dict[str, Any] = {"expect_checkpoint_sha256": "c" * 64, "expect_labels": LABELS}
    with pytest.raises(CalibrationError, match=message):
        load_calibration(path, **{**expectations, **kwargs})


def test_unreadable_calibration_file(tmp_path: Path) -> None:
    path = tmp_path / "c.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CalibrationError, match="unreadable"):
        load_calibration(path, expect_checkpoint_sha256="c" * 64, expect_labels=LABELS)
