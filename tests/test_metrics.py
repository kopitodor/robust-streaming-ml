"""Metric fixtures computed by hand (see the comments for each expected value)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from robust_streaming_ml.evaluation.metrics import (
    accuracy,
    classification_metrics,
    confusion_matrix,
    expected_calibration_error,
    log_softmax,
    negative_log_likelihood,
    per_class_scores,
    predictions_from_logits,
    softmax,
)

Y_TRUE = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
Y_PRED = np.array([0, 1, 1, 1, 2, 0], dtype=np.int64)


def test_confusion_matrix_rows_are_truth_columns_are_predictions() -> None:
    assert confusion_matrix(Y_TRUE, Y_PRED, 3).tolist() == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    # A class that never occurs still gets a row and column (fixed label order).
    assert confusion_matrix(np.array([0]), np.array([0]), 3).tolist() == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]


def test_accuracy() -> None:
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(4 / 6)
    with pytest.raises(ValueError):
        accuracy(np.array([], dtype=np.int64), np.array([], dtype=np.int64))


def test_per_class_and_macro_f1() -> None:
    # class 0: TP=1 FP=1 FN=1 -> P=0.5, R=0.5, F1=0.5
    # class 1: TP=2 FP=1 FN=0 -> P=2/3, R=1, F1=0.8
    # class 2: TP=1 FP=0 FN=1 -> P=1, R=0.5, F1=2/3
    scores = per_class_scores(confusion_matrix(Y_TRUE, Y_PRED, 3))
    np.testing.assert_allclose(scores["precision"], [0.5, 2 / 3, 1.0])
    np.testing.assert_allclose(scores["recall"], [0.5, 1.0, 0.5])
    np.testing.assert_allclose(scores["f1"], [0.5, 0.8, 2 / 3])
    assert scores["support"].tolist() == [2, 2, 2]
    assert float(np.mean(scores["f1"])) == pytest.approx((0.5 + 0.8 + 2 / 3) / 3)


def test_zero_division_policy_gives_zero() -> None:
    # truth [0, 1], predictions [0, 0], 3 classes:
    # class 0: P=1/2, R=1 -> F1=2/3; class 1: P=0/0 -> 0, R=0 -> F1 0;
    # class 2 absent from truth and predictions: everything 0/0 -> 0.
    scores = per_class_scores(confusion_matrix(np.array([0, 1]), np.array([0, 0]), 3))
    np.testing.assert_allclose(scores["precision"], [0.5, 0.0, 0.0])
    np.testing.assert_allclose(scores["recall"], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(scores["f1"], [2 / 3, 0.0, 0.0])
    assert float(np.mean(scores["f1"])) == pytest.approx(2 / 9)


def test_nll_from_known_probabilities() -> None:
    probs = np.array([[0.5, 0.25, 0.25], [0.1, 0.8, 0.1]])
    log_probs = log_softmax(np.log(probs))  # softmax(log p) = p
    # -(ln 0.5 + ln 0.8) / 2 = (0.693147 + 0.223144) / 2 = 0.458145
    assert negative_log_likelihood(log_probs, np.array([0, 1])) == pytest.approx(
        -(math.log(0.5) + math.log(0.8)) / 2
    )
    assert negative_log_likelihood(log_probs, np.array([0, 1])) == pytest.approx(0.458145, abs=1e-6)


def test_ece_hand_computed_with_five_bins() -> None:
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.3, 0.7], [0.55, 0.45]])
    y = np.array([0, 1, 1, 0])
    # bin 4 [0.8, 1.0]: 2 examples, accuracy 0.5, confidence 0.9 -> 2/4 * 0.4 = 0.2
    # bin 3 [0.6, 0.8): 1 example, accuracy 1, confidence 0.7 -> 1/4 * 0.3 = 0.075
    # bin 2 [0.4, 0.6): 1 example, accuracy 1, confidence 0.55 -> 1/4 * 0.45 = 0.1125
    ece, bins = expected_calibration_error(probs, y, n_bins=5)
    assert ece == pytest.approx(0.3875)
    assert [b["count"] for b in bins] == [0, 0, 1, 1, 2]
    assert bins[4]["accuracy"] == 0.5 and bins[4]["mean_confidence"] == pytest.approx(0.9)
    assert bins[0]["accuracy"] is None and bins[0]["mean_confidence"] is None
    assert (bins[4]["lower"], bins[4]["upper"]) == (0.8, 1.0)


def test_ece_binning_edges_and_perfect_calibration() -> None:
    # Confidence 1.0 falls in the last bin; 0.5 falls in bin floor(0.5 * 5) = 2.
    _, bins = expected_calibration_error(np.array([[1.0, 0.0], [0.5, 0.5]]), np.array([0, 0]), 5)
    assert [b["count"] for b in bins] == [0, 0, 1, 0, 1]
    ece, _ = expected_calibration_error(np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([0, 1]), 15)
    assert ece == 0.0


def test_softmax_temperature_and_argmax_ties() -> None:
    logits = np.array([[2.0, 0.0, 2.0], [1.0, 3.0, -1.0]])
    np.testing.assert_allclose(softmax(logits).sum(axis=1), 1.0)
    np.testing.assert_allclose(softmax(logits, 2.0), softmax(logits / 2.0))
    assert predictions_from_logits(logits).tolist() == [0, 1]  # tie -> lowest index
    with pytest.raises(ValueError):
        softmax(logits, 0.0)
    big = np.array([[1000.0, 0.0]])
    assert np.isfinite(log_softmax(big)).all()


def test_classification_metrics_bundle_and_temperature_invariance() -> None:
    logits = np.log(np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]))
    y = np.array([0, 1, 1, 2])
    labels = ("a", "b", "c")
    raw = classification_metrics(logits, y, labels)
    hot = classification_metrics(logits, y, labels, temperature=0.5)
    assert raw["accuracy"] == pytest.approx(0.75) and raw["examples"] == 4
    assert raw["confusion_matrix"]["rows_true_cols_pred"] == [[1, 0, 0], [1, 1, 0], [0, 0, 1]]
    assert raw["nll"] == pytest.approx(
        -(math.log(0.7) + math.log(0.5) + math.log(0.3) + math.log(0.8)) / 4
    )
    for key in ("accuracy", "macro_f1", "confusion_matrix", "per_class"):
        assert raw[key] == hot[key]  # temperature cannot change the argmax
    assert raw["nll"] != hot["nll"]
    with pytest.raises(ValueError, match="logits must be"):
        classification_metrics(logits[:3], y, labels)


def test_invalid_labels_are_rejected() -> None:
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 3]), np.array([0, 0]), 3)
    with pytest.raises(ValueError):
        confusion_matrix(np.array([0, 1]), np.array([0]), 3)
