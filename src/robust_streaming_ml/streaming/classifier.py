"""Per-window prediction and causal aggregation (``kw10-streaming-v1``, §2.4-2.6).

Each emitted window is inferred on its own (batch size 1) by a backend that
maps ``[1, W]`` float32 windows to ``[1, C]`` float32 logits. The window's raw
probabilities are ``softmax(logits)`` in float64 (``T = 1``), its label is the
argmax of the logits (ties to the lowest index), and its confidence is the
largest probability.

The aggregate of window ``k`` is the mean of the probability vectors of
windows ``max(0, k - M + 1) .. k``. It is computed once, when window ``k`` is
emitted, from past windows only. The stream decision is the aggregate of the
last emitted window after ``flush``; an empty stream has no decision.

:class:`StreamingClassifier` runs this on pushed chunks; :func:`classify_offline`
runs it on :func:`~robust_streaming_ml.streaming.windowing.offline_windows`
of a complete signal. Both share :class:`WindowPredictor`, so any chunking
yields bit-identical outputs.
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import torch

from robust_streaming_ml.evaluation.metrics import predictions_from_logits, softmax
from robust_streaming_ml.features import LogMelSpectrogram
from robust_streaming_ml.model import KeywordCNN
from robust_streaming_ml.streaming.windowing import (
    StreamingError,
    StreamingSpec,
    StreamingWindower,
    Window,
    offline_windows,
)

__all__ = [
    "StreamDecision",
    "StreamingClassifier",
    "TorchWindowBackend",
    "WindowBackend",
    "WindowPrediction",
    "WindowPredictor",
    "classify_offline",
    "classify_stream",
]

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]


class WindowBackend(Protocol):
    """Maps float32 windows ``[B, W]`` to float32 logits ``[B, C]``."""

    name: str

    def __call__(self, windows: Float32) -> Float32: ...


class TorchWindowBackend:
    """The PyTorch reference path: the checkpoint's front end and model in eval mode."""

    name = "pytorch"

    def __init__(self, model: KeywordCNN, frontend: LogMelSpectrogram) -> None:
        self.model = model.eval()
        self.frontend = frontend

    def __call__(self, windows: Float32) -> Float32:
        # Windows are read-only; torch needs a writable array.
        batch = torch.from_numpy(np.array(windows, dtype=np.float32, order="C", copy=True))
        with torch.no_grad():
            logits = self.model(self.frontend(batch))
        return np.asarray(logits.to(torch.float32).numpy().copy(), dtype=np.float32)


@dataclass(frozen=True)
class WindowPrediction:
    """Outputs for one window, available as soon as the window is emitted."""

    window: Window
    logits: Float32
    probabilities: Float64
    label_index: int
    confidence: float
    aggregate_probabilities: Float64
    aggregate_label_index: int
    aggregate_confidence: float
    aggregate_windows: int
    inference_ns: int | None = None

    def describe(self, labels: Sequence[str], sample_rate_hz: int) -> dict[str, Any]:
        return {
            **self.window.timing(sample_rate_hz),
            "label": labels[self.label_index],
            "confidence": self.confidence,
            "aggregate_label": labels[self.aggregate_label_index],
            "aggregate_confidence": self.aggregate_confidence,
            "aggregate_windows": self.aggregate_windows,
            "inference_ms": None if self.inference_ns is None else self.inference_ns / 1e6,
        }


@dataclass(frozen=True)
class StreamDecision:
    """The stream (clip) decision: the aggregate of the last emitted window."""

    label_index: int
    label: str
    confidence: float
    probabilities: Float64
    windows: int
    last_window_index: int


class WindowPredictor:
    """Batch-1 inference, raw probabilities, and the causal aggregate, window by window."""

    def __init__(
        self,
        backend: WindowBackend,
        spec: StreamingSpec,
        labels: Sequence[str],
        *,
        measure_time: bool = False,
    ) -> None:
        self.backend = backend
        self.spec = spec
        self.labels = tuple(labels)
        self.measure_time = measure_time
        self._recent: deque[Float64] = deque(maxlen=spec.aggregation_windows)
        self._last: WindowPrediction | None = None

    def reset(self) -> None:
        self._recent.clear()
        self._last = None

    @property
    def last(self) -> WindowPrediction | None:
        return self._last

    def _infer(self, window: Window) -> tuple[Float32, int | None]:
        batch = window.samples.reshape(1, -1)
        started = time.perf_counter_ns() if self.measure_time else 0
        logits = self.backend(batch)
        elapsed = time.perf_counter_ns() - started if self.measure_time else None
        num_classes = len(self.labels)
        if not isinstance(logits, np.ndarray) or logits.dtype != np.float32:
            raise StreamingError(f"backend {self.backend.name} must return float32 logits")
        if logits.shape != (1, num_classes):
            raise StreamingError(
                f"backend {self.backend.name} returned shape {logits.shape}, "
                f"expected (1, {num_classes})"
            )
        if not np.isfinite(logits).all():
            raise StreamingError(f"backend {self.backend.name} returned non-finite logits")
        return logits, elapsed

    def predict(self, window: Window) -> WindowPrediction:
        if window.samples.shape != (self.spec.window_samples,):
            raise StreamingError(f"window has shape {window.samples.shape}")
        logits, elapsed = self._infer(window)
        probabilities = softmax(logits)[0]
        self._recent.append(probabilities)
        aggregate = np.stack(list(self._recent)).mean(axis=0)
        aggregate_label = int(np.argmax(aggregate))
        prediction = WindowPrediction(
            window=window,
            logits=logits[0].copy(),
            probabilities=probabilities,
            label_index=int(predictions_from_logits(logits)[0]),
            confidence=float(probabilities.max()),
            aggregate_probabilities=aggregate,
            aggregate_label_index=aggregate_label,
            aggregate_confidence=float(aggregate[aggregate_label]),
            aggregate_windows=len(self._recent),
            inference_ns=elapsed,
        )
        self._last = prediction
        return prediction

    def decision(self) -> StreamDecision | None:
        last = self._last
        if last is None:
            return None
        return StreamDecision(
            label_index=last.aggregate_label_index,
            label=self.labels[last.aggregate_label_index],
            confidence=last.aggregate_confidence,
            probabilities=last.aggregate_probabilities,
            windows=last.window.index + 1,
            last_window_index=last.window.index,
        )


class StreamingClassifier:
    """Chunked streaming inference: windower + per-window predictor."""

    def __init__(
        self,
        backend: WindowBackend,
        spec: StreamingSpec,
        labels: Sequence[str],
        *,
        sample_rate_hz: int,
        measure_time: bool = False,
    ) -> None:
        self.spec = spec
        self.windower = StreamingWindower(spec, sample_rate_hz=sample_rate_hz)
        self.predictor = WindowPredictor(backend, spec, labels, measure_time=measure_time)

    @property
    def labels(self) -> tuple[str, ...]:
        return self.predictor.labels

    def push(self, chunk: Any) -> list[WindowPrediction]:
        return [self.predictor.predict(w) for w in self.windower.push(chunk)]

    def flush(self) -> list[WindowPrediction]:
        return [self.predictor.predict(w) for w in self.windower.flush()]

    def reset(self) -> None:
        self.windower.reset()
        self.predictor.reset()

    def decision(self) -> StreamDecision | None:
        """The stream decision; only final once :meth:`flush` has been called."""
        if not self.windower.finished:
            raise StreamingError("the stream decision is defined after flush()")
        return self.predictor.decision()


def classify_offline(
    samples: Any,
    backend: WindowBackend,
    spec: StreamingSpec,
    labels: Sequence[str],
) -> tuple[list[WindowPrediction], StreamDecision | None]:
    """The offline reference: all windows of a complete signal, then the same predictor."""
    predictor = WindowPredictor(backend, spec, labels)
    predictions = [predictor.predict(w) for w in offline_windows(samples, spec)]
    return predictions, predictor.decision()


def classify_stream(
    chunks: Iterable[Any],
    backend: WindowBackend,
    spec: StreamingSpec,
    labels: Sequence[str],
    *,
    sample_rate_hz: int,
) -> tuple[list[WindowPrediction], StreamDecision | None]:
    """Push every chunk, flush, and return all window outputs and the stream decision."""
    classifier = StreamingClassifier(backend, spec, labels, sample_rate_hz=sample_rate_hz)
    predictions: list[WindowPrediction] = []
    for chunk in chunks:
        predictions.extend(classifier.push(chunk))
    predictions.extend(classifier.flush())
    return predictions, classifier.decision()
