"""Temperature scaling fitted on validation predictions only (protocol section 6).

``T > 0`` rescales logits: ``softmax(logits / T)``. It is found by minimising
the validation NLL over ``beta = 1/T``. The NLL is convex in ``beta`` because
log-sum-exp is convex and the true-class term is linear. The search is a
deterministic golden-section search over ``beta`` in ``[0.05, 20]`` with a
fixed number of iterations; the final ``beta`` is the midpoint of the last
interval.

The fit refuses any predictions that are not from the validation split.
``calibration.json`` records the parameters, the fitting inputs, and the
compatibility metadata; :func:`load_calibration` verifies them.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from robust_streaming_ml.evaluation.metrics import log_softmax, negative_log_likelihood
from robust_streaming_ml.evaluation.predictions import PredictionSet

__all__ = [
    "BETA_BOUNDS",
    "CALIBRATION_FORMAT",
    "CALIBRATION_FORMAT_VERSION",
    "FIT_SPLIT",
    "SEARCH_ITERATIONS",
    "CalibrationError",
    "TemperatureFit",
    "calibration_record",
    "fit_temperature",
    "load_calibration",
    "save_calibration",
]

CALIBRATION_FORMAT = "robust-streaming-ml/calibration"
CALIBRATION_FORMAT_VERSION = 1
FIT_SPLIT = "validation"
BETA_BOUNDS = (0.05, 20.0)
SEARCH_ITERATIONS = 200
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0


class CalibrationError(RuntimeError):
    """Raised on a forbidden fit or an incompatible calibration file."""


@dataclass(frozen=True)
class TemperatureFit:
    temperature: float
    inverse_temperature: float
    nll_before: float
    nll_after: float
    at_bound: bool
    examples: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "inverse_temperature": self.inverse_temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "at_bound": self.at_bound,
            "examples": self.examples,
        }


def fit_temperature(
    predictions: PredictionSet,
    *,
    bounds: tuple[float, float] = BETA_BOUNDS,
    iterations: int = SEARCH_ITERATIONS,
) -> TemperatureFit:
    """Fit ``T`` on validation predictions (any other split is refused)."""
    if predictions.split != FIT_SPLIT:
        raise CalibrationError(
            f"calibration may only be fitted on {FIT_SPLIT!r} predictions, "
            f"got {predictions.split!r}"
        )
    if len(predictions) == 0:
        raise CalibrationError("no predictions to fit on")
    lo, hi = bounds
    if not 0 < lo < 1 < hi:
        raise CalibrationError("beta bounds must satisfy 0 < lo < 1 < hi")
    y = predictions.y_true
    logits = predictions.logits

    def nll(beta: float) -> float:
        return negative_log_likelihood(log_softmax(logits, 1.0 / beta), y)

    a, b = lo, hi
    c = b - _GOLDEN * (b - a)
    d = a + _GOLDEN * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = nll(d)
    beta = (a + b) / 2.0
    at_bound = math.isclose(beta, lo, rel_tol=1e-6) or math.isclose(beta, hi, rel_tol=1e-6)
    return TemperatureFit(
        temperature=1.0 / beta,
        inverse_temperature=beta,
        nll_before=nll(1.0),
        nll_after=nll(beta),
        at_bound=at_bound,
        examples=len(predictions),
    )


def calibration_record(
    fit: TemperatureFit,
    *,
    fitted_on: Mapping[str, Any],
    compatibility: Mapping[str, Any],
) -> dict[str, Any]:
    """The JSON document persisted as ``calibration.json``."""
    return {
        "format": CALIBRATION_FORMAT,
        "format_version": CALIBRATION_FORMAT_VERSION,
        "method": "temperature_scaling",
        "parameters": fit.as_dict(),
        "search": {
            "variable": "inverse_temperature",
            "objective": "mean validation NLL",
            "method": "golden_section",
            "bounds": list(BETA_BOUNDS),
            "iterations": SEARCH_ITERATIONS,
        },
        "fitted_on": {"split": FIT_SPLIT, **dict(fitted_on)},
        "compatibility": dict(compatibility),
    }


def save_calibration(path: Path, record: Mapping[str, Any]) -> None:
    if record.get("fitted_on", {}).get("split") != FIT_SPLIT:
        raise CalibrationError("refusing to save a calibration not fitted on validation")
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_calibration(
    path: Path, *, expect_checkpoint_sha256: str, expect_labels: Sequence[str]
) -> dict[str, Any]:
    """Load ``calibration.json`` and verify its format, fit split, and compatibility."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise CalibrationError(f"{path}: unreadable calibration: {exc}") from exc
    if not isinstance(record, dict):
        raise CalibrationError(f"{path}: calibration must be a JSON object")
    if (record.get("format"), record.get("format_version")) != (
        CALIBRATION_FORMAT,
        CALIBRATION_FORMAT_VERSION,
    ):
        raise CalibrationError(f"{path}: unsupported calibration format")
    if record.get("method") != "temperature_scaling":
        raise CalibrationError(f"{path}: unsupported method {record.get('method')!r}")
    if record.get("fitted_on", {}).get("split") != FIT_SPLIT:
        raise CalibrationError(f"{path}: calibration was not fitted on {FIT_SPLIT!r}")
    compat = record.get("compatibility", {})
    if compat.get("checkpoint_sha256") != expect_checkpoint_sha256:
        raise CalibrationError(f"{path}: calibration belongs to another checkpoint")
    if compat.get("labels") != list(expect_labels):
        raise CalibrationError(f"{path}: calibration labels do not match")
    params = record.get("parameters", {})
    temperature = params.get("temperature")
    if not isinstance(temperature, float) or not temperature > 0:
        raise CalibrationError(f"{path}: invalid temperature {temperature!r}")
    if not math.isclose(temperature * params.get("inverse_temperature", 0.0), 1.0, rel_tol=1e-12):
        raise CalibrationError(f"{path}: temperature and inverse_temperature disagree")
    return record
