"""PyTorch-versus-ONNX parity with predeclared tolerances (``kw10-onnx-v1``, §4.3).

Per window and class:

- ``|onnx logit - torch logit| <= LOGIT_ATOL`` (1e-2);
- ``|onnx prob - torch prob| <= PROBABILITY_ATOL`` (1e-3), raw float64 softmax;
- aggregated probabilities within ``AGGREGATE_ATOL`` (1e-3);
- the top-1 label agrees unless the reference top-2 logit margin is at most
  ``2 * LOGIT_ATOL`` (only then is a flip consistent with the logit tolerance);
  such near-tie flips are counted, not violations. The aggregated top-1 label
  is treated the same way with ``2 * AGGREGATE_ATOL``.

The functions are pure and work on arrays, so every statistic in a parity
report can be regenerated from the stored logits.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.evaluation.metrics import predictions_from_logits, softmax

__all__ = [
    "AGGREGATE_ATOL",
    "LOGIT_ATOL",
    "MAX_LISTED_VIOLATIONS",
    "PROBABILITY_ATOL",
    "causal_aggregates",
    "compare_aggregates",
    "compare_logits",
    "tolerances",
]

LOGIT_ATOL = 1e-2
PROBABILITY_ATOL = 1e-3
AGGREGATE_ATOL = 1e-3
MAX_LISTED_VIOLATIONS = 100

Float = npt.NDArray[np.floating[Any]]


def tolerances() -> dict[str, float]:
    return {
        "logit_atol": LOGIT_ATOL,
        "probability_atol": PROBABILITY_ATOL,
        "aggregate_probability_atol": AGGREGATE_ATOL,
        "top1_near_tie_logit_margin": 2 * LOGIT_ATOL,
        "aggregate_top1_near_tie_margin": 2 * AGGREGATE_ATOL,
    }


def _top2_margin(values: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    ordered = np.sort(values, axis=1)
    margin: npt.NDArray[np.float64] = ordered[:, -1] - ordered[:, -2]
    return margin


def _summary(diff: npt.NDArray[np.float64]) -> dict[str, float]:
    per_row = diff.max(axis=1) if diff.size else np.zeros(0)
    return {
        "max": float(per_row.max()) if per_row.size else 0.0,
        "mean": float(diff.mean()) if diff.size else 0.0,
        "p99_row_max": float(np.quantile(per_row, 0.99)) if per_row.size else 0.0,
    }


def _check_shapes(reference: Float, candidate: Float) -> None:
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise ValueError(f"shapes differ: {reference.shape} vs {candidate.shape}")
    if not (np.isfinite(reference).all() and np.isfinite(candidate).all()):
        raise ValueError("parity inputs contain non-finite values")


def compare_logits(
    reference: npt.NDArray[np.float32],
    candidate: npt.NDArray[np.float32],
    *,
    logit_atol: float = LOGIT_ATOL,
    probability_atol: float = PROBABILITY_ATOL,
    row_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Per-window logit and probability parity of ``candidate`` against ``reference``."""
    _check_shapes(reference, candidate)
    ref64 = reference.astype(np.float64)
    cand64 = candidate.astype(np.float64)
    logit_diff = np.abs(cand64 - ref64)
    prob_diff = np.abs(softmax(candidate) - softmax(reference))
    ref_top = predictions_from_logits(reference)
    cand_top = predictions_from_logits(candidate)
    margin = _top2_margin(ref64) if reference.shape[1] > 1 else np.full(len(ref64), np.inf)
    flips = ref_top != cand_top
    near_tie = margin <= 2 * logit_atol
    violations: list[dict[str, Any]] = []
    counts = {"logit": 0, "probability": 0, "top1": 0}
    for i in range(reference.shape[0]):
        kinds = []
        if logit_diff[i].max() > logit_atol:
            kinds.append("logit")
        if prob_diff[i].max() > probability_atol:
            kinds.append("probability")
        if flips[i] and not near_tie[i]:
            kinds.append("top1")
        for kind in kinds:
            counts[kind] += 1
        if kinds and len(violations) < MAX_LISTED_VIOLATIONS:
            violations.append(
                {
                    "row": i,
                    "id": row_ids[i] if row_ids is not None else None,
                    "kinds": kinds,
                    "max_abs_logit_diff": float(logit_diff[i].max()),
                    "max_abs_probability_diff": float(prob_diff[i].max()),
                    "reference_top1": int(ref_top[i]),
                    "candidate_top1": int(cand_top[i]),
                    "reference_top2_margin": float(margin[i]),
                }
            )
    violating_rows = int(
        np.sum(
            (logit_diff.max(axis=1, initial=0.0) > logit_atol)
            | (prob_diff.max(axis=1, initial=0.0) > probability_atol)
            | (flips & ~near_tie)
        )
    )
    return {
        "windows": int(reference.shape[0]),
        "abs_logit_diff": _summary(logit_diff),
        "abs_probability_diff": _summary(prob_diff),
        "top1_disagreements": int(flips.sum()),
        "top1_near_tie_flips": int((flips & near_tie).sum()),
        "violations_by_kind": counts,
        "violating_windows": violating_rows,
        "violations": violations,
        "passed": violating_rows == 0,
    }


def causal_aggregates(probabilities: npt.NDArray[np.float64], size: int) -> npt.NDArray[np.float64]:
    """Mean of the last ``size`` probability vectors at each window (one stream, in order)."""
    out = np.zeros_like(probabilities, dtype=np.float64)
    for k in range(probabilities.shape[0]):
        out[k] = probabilities[max(0, k - size + 1) : k + 1].mean(axis=0)
    return out


def compare_aggregates(
    reference: npt.NDArray[np.float64],
    candidate: npt.NDArray[np.float64],
    *,
    atol: float = AGGREGATE_ATOL,
    row_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Parity of aggregated probability vectors and their top-1 labels."""
    _check_shapes(reference, candidate)
    diff = np.abs(candidate - reference)
    ref_top = np.argmax(reference, axis=1)
    cand_top = np.argmax(candidate, axis=1)
    margin = _top2_margin(reference) if reference.shape[1] > 1 else np.full(len(diff), np.inf)
    flips = ref_top != cand_top
    near_tie = margin <= 2 * atol
    bad = (diff.max(axis=1, initial=0.0) > atol) | (flips & ~near_tie)
    violations = [
        {
            "row": int(i),
            "id": row_ids[int(i)] if row_ids is not None else None,
            "max_abs_diff": float(diff[i].max()),
            "reference_top1": int(ref_top[i]),
            "candidate_top1": int(cand_top[i]),
            "reference_top2_margin": float(margin[i]),
        }
        for i in np.flatnonzero(bad)[:MAX_LISTED_VIOLATIONS]
    ]
    return {
        "windows": int(reference.shape[0]),
        "abs_diff": _summary(diff),
        "top1_disagreements": int(flips.sum()),
        "top1_near_tie_flips": int((flips & near_tie).sum()),
        "violating_windows": int(bad.sum()),
        "violations": violations,
        "passed": not bool(bad.any()),
    }
