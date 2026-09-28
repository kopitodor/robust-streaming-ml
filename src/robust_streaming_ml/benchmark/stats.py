"""Latency statistics for ``kw10-latency-v1`` (docs/benchmark-protocol.md, §3).

Percentiles use the nearest-rank method on integer nanoseconds: for ``N``
sorted values ``x(1) <= ... <= x(N)``, ``p_q = x(ceil(q * N / 100))``, with
integer arithmetic and no interpolation.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

__all__ = ["PERCENTILES", "nearest_rank", "summarize"]

PERCENTILES = (50, 95, 99)


def nearest_rank(sorted_ns: Sequence[int], percent: int) -> int:
    """The ``percent``-th percentile (1-100) of sorted values, by nearest rank."""
    if not sorted_ns:
        raise ValueError("no values")
    if not 1 <= percent <= 100:
        raise ValueError("percent must be in [1, 100]")
    if any(a > b for a, b in pairwise(sorted_ns)):
        raise ValueError("values must be sorted")
    rank = (percent * len(sorted_ns) + 99) // 100  # ceil(percent * N / 100)
    return int(sorted_ns[rank - 1])


def summarize(durations_ns: Sequence[int]) -> dict[str, Any]:
    """Count, nearest-rank p50/p95/p99, mean, min, and max, in milliseconds."""
    values = sorted(int(v) for v in durations_ns)
    if not values:
        raise ValueError("no durations")
    if values[0] < 0:
        raise ValueError("durations must be non-negative")
    summary: dict[str, Any] = {"n": len(values)}
    for percent in PERCENTILES:
        summary[f"p{percent}_ms"] = nearest_rank(values, percent) / 1e6
    summary["mean_ms"] = sum(values) / len(values) / 1e6
    summary["min_ms"] = values[0] / 1e6
    summary["max_ms"] = values[-1] / 1e6
    return summary
