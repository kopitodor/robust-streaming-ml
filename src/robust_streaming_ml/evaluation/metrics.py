"""Classification and calibration metrics (docs/evaluation-protocol.md, section 5).

All functions take plain NumPy arrays, compute in float64, and are pure.
Label order is fixed by the caller: class ``i`` is row/column ``i``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_ECE_BINS",
    "accuracy",
    "classification_metrics",
    "confusion_matrix",
    "expected_calibration_error",
    "log_softmax",
    "negative_log_likelihood",
    "per_class_scores",
    "predictions_from_logits",
    "softmax",
]

DEFAULT_ECE_BINS = 15
IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]


def _check_labels(y_true: IntArray, num_classes: int) -> None:
    if y_true.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if y_true.size and (y_true.min() < 0 or y_true.max() >= num_classes):
        raise ValueError(f"labels must be in [0, {num_classes})")


def log_softmax(logits: npt.ArrayLike, temperature: float = 1.0) -> FloatArray:
    """Row-wise ``log softmax(logits / temperature)`` in float64 (max-subtracted)."""
    if not temperature > 0:
        raise ValueError("temperature must be positive")
    z = np.asarray(logits, dtype=np.float64) / temperature
    if z.ndim != 2:
        raise ValueError("logits must be [examples, classes]")
    shifted = z - z.max(axis=1, keepdims=True)
    result: FloatArray = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    return result


def softmax(logits: npt.ArrayLike, temperature: float = 1.0) -> FloatArray:
    return np.exp(log_softmax(logits, temperature))


def predictions_from_logits(logits: npt.ArrayLike) -> IntArray:
    """Argmax per row; ties go to the lowest class index."""
    return np.asarray(np.argmax(np.asarray(logits, dtype=np.float64), axis=1), dtype=np.int64)


def confusion_matrix(y_true: IntArray, y_pred: IntArray, num_classes: int) -> IntArray:
    """``C[i, j]`` = count of examples with true class ``i`` predicted as ``j``."""
    _check_labels(y_true, num_classes)
    _check_labels(y_pred, num_classes)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def accuracy(y_true: IntArray, y_pred: IntArray) -> float:
    if y_true.size == 0:
        raise ValueError("accuracy of an empty set is undefined")
    return float(np.mean(y_true == y_pred))


def _safe_divide(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """Element-wise division where 0/0 (or x/0) is defined as 0 (zero-division policy)."""
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=denominator != 0)
    return out


def per_class_scores(matrix: IntArray) -> dict[str, FloatArray | IntArray]:
    """Precision, recall, F1 (zero-division -> 0), and support per class."""
    m = matrix.astype(np.float64)
    tp = np.diag(m)
    predicted = m.sum(axis=0)
    actual = m.sum(axis=1)
    precision = _safe_divide(tp, predicted)
    recall = _safe_divide(tp, actual)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": matrix.sum(axis=1).astype(np.int64),
    }


def negative_log_likelihood(log_probs: FloatArray, y_true: IntArray) -> float:
    """Mean ``-log p(true class)`` (natural log) from log-probabilities."""
    _check_labels(y_true, log_probs.shape[1])
    if y_true.size == 0:
        raise ValueError("NLL of an empty set is undefined")
    return float(-np.mean(log_probs[np.arange(y_true.size), y_true]))


def expected_calibration_error(
    probs: FloatArray, y_true: IntArray, n_bins: int = DEFAULT_ECE_BINS
) -> tuple[float, list[dict[str, Any]]]:
    """Top-label ECE with ``n_bins`` equal-width bins.

    Confidence ``c`` goes to bin ``min(floor(c * n_bins), n_bins - 1)``: bins are
    ``[k/n, (k+1)/n)`` and the last includes 1.0. Returns the ECE and one entry
    per bin (``count``, ``accuracy``, ``mean_confidence``; the last two are
    ``None`` for empty bins).
    """
    if n_bins < 1:
        raise ValueError("n_bins must be positive")
    _check_labels(y_true, probs.shape[1])
    n = y_true.size
    if n == 0:
        raise ValueError("ECE of an empty set is undefined")
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    index = np.minimum(np.floor(confidence * n_bins).astype(np.int64), n_bins - 1)
    ece = 0.0
    bins: list[dict[str, Any]] = []
    for k in range(n_bins):
        mask = index == k
        count = int(mask.sum())
        entry: dict[str, Any] = {
            "bin": k,
            "lower": k / n_bins,
            "upper": (k + 1) / n_bins,
            "count": count,
            "accuracy": None,
            "mean_confidence": None,
        }
        if count:
            acc = float(correct[mask].mean())
            conf = float(confidence[mask].mean())
            entry["accuracy"] = acc
            entry["mean_confidence"] = conf
            ece += count / n * abs(acc - conf)
        bins.append(entry)
    return ece, bins


def classification_metrics(
    logits: npt.ArrayLike,
    y_true: IntArray,
    labels: tuple[str, ...] | list[str],
    *,
    temperature: float = 1.0,
    n_bins: int = DEFAULT_ECE_BINS,
) -> dict[str, Any]:
    """All protocol metrics for one set of logits at a given temperature."""
    num_classes = len(labels)
    logits64 = np.asarray(logits, dtype=np.float64)
    if logits64.shape != (y_true.size, num_classes):
        raise ValueError(f"logits must be [{y_true.size}, {num_classes}], got {logits64.shape}")
    y_pred = predictions_from_logits(logits64)
    log_probs = log_softmax(logits64, temperature)
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    scores = per_class_scores(matrix)
    ece, bins = expected_calibration_error(np.exp(log_probs), y_true, n_bins)
    return {
        "examples": int(y_true.size),
        "temperature": temperature,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": float(np.mean(scores["f1"])),
        "nll": negative_log_likelihood(log_probs, y_true),
        "ece": ece,
        "ece_bins": n_bins,
        "per_class": {
            label: {
                "precision": float(scores["precision"][i]),
                "recall": float(scores["recall"][i]),
                "f1": float(scores["f1"][i]),
                "support": int(scores["support"][i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": {"labels": list(labels), "rows_true_cols_pred": matrix.tolist()},
        "calibration_bins": bins,
    }
