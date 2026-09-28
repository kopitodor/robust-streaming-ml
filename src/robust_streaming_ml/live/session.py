"""One live-microphone stream: audio contract, ordering, resampling, streaming inference.

**Browser-to-backend audio contract** (docs/live-demo.md):

- A session is created for one device sample rate ``R`` from
  :data:`~robust_streaming_ml.live.resample.SUPPORTED_INPUT_RATES`.
- Each chunk is a request body of raw little-endian IEEE float32 samples
  (``f32le``): mono, at rate ``R``, the mean of the device channels. It must hold
  1 to ``2 * R`` samples (at most 2 s), all finite. Anything else is rejected
  with :class:`AudioContractError` and the session is unchanged.
- Chunks carry a sequence number that must equal the next expected one
  (0, 1, 2, ...). Duplicates, gaps, or reordering are rejected with
  :class:`ChunkOrderError`, and the session is unchanged.

Each accepted chunk goes through
:class:`~robust_streaming_ml.live.resample.StreamingResampler` (16 kHz int16),
then through the existing ``kw10-streaming-v1``
:class:`~robust_streaming_ml.streaming.classifier.StreamingClassifier`. Its
windows are returned with their timestamps, labels, confidences, causal
aggregate (mean of the last 5 windows), and inference time.

``stop`` ends the stream: the resampler and classifier are flushed, and the
protocol's stream decision is returned as the **final** decision. ``reset``
returns the session to its initial state (sequence number 0).

**Stable indicator** (display rule; it does not change any prediction): a
window's aggregate is marked *stable* only when it uses the full 5-window
span and all 5 windows' own top-1 labels equal the aggregate label.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.live.resample import StreamingResampler
from robust_streaming_ml.streaming.classifier import (
    StreamingClassifier,
    WindowBackend,
    WindowPrediction,
)
from robust_streaming_ml.streaming.windowing import StreamingSpec

__all__ = [
    "MAX_CHUNK_SECONDS",
    "STABLE_RULE",
    "AudioContractError",
    "ChunkOrderError",
    "LiveSession",
    "SessionStateError",
    "decode_chunk",
]

MAX_CHUNK_SECONDS = 2
STABLE_RULE = (
    "stable = the aggregate uses the full 5-window span and all 5 windows' top-1 labels equal "
    "the aggregate label (display rule only; predictions are unchanged)"
)

Float32 = npt.NDArray[np.float32]


class AudioContractError(ValueError):
    """A chunk violates the browser-to-backend audio contract."""


class ChunkOrderError(ValueError):
    """A chunk arrived out of order (duplicate, gap, or reordering)."""


class SessionStateError(ValueError):
    """The request does not fit the session's state (e.g. audio after stop)."""


def decode_chunk(body: bytes, *, sample_rate_hz: int) -> Float32:
    """Raw ``f32le`` mono samples, validated against the contract."""
    if not body:
        raise AudioContractError("empty chunk")
    if len(body) % 4:
        raise AudioContractError(
            f"chunk has {len(body)} bytes, not a whole number of float32 samples"
        )
    samples = np.frombuffer(body, dtype="<f4").astype(np.float32)
    limit = MAX_CHUNK_SECONDS * sample_rate_hz
    if samples.size > limit:
        raise AudioContractError(f"chunk has {samples.size} samples; the limit is {limit} (2 s)")
    if not np.isfinite(samples).all():
        raise AudioContractError("chunk contains non-finite samples")
    return samples


def _level_dbfs(prediction: WindowPrediction) -> float | None:
    window = prediction.window
    valid = window.samples[: window.valid_samples].astype(np.float64)
    rms = math.sqrt(float(np.mean(valid * valid))) if valid.size else 0.0
    return None if rms == 0.0 else 20.0 * math.log10(rms)


class LiveSession:
    """Ordered chunks -> 16 kHz int16 -> kw10-streaming-v1 windows."""

    def __init__(
        self,
        backend: WindowBackend,
        spec: StreamingSpec,
        labels: Sequence[str],
        *,
        input_rate: int,
    ) -> None:
        self.spec = spec
        self.labels = tuple(labels)
        self.input_rate = input_rate
        self.resampler = StreamingResampler(input_rate)
        self.classifier = StreamingClassifier(
            backend, spec, self.labels, sample_rate_hz=spec.sample_rate_hz, measure_time=True
        )
        self._recent: deque[int] = deque(maxlen=spec.aggregation_windows)
        self.next_seq = 0
        self.finished = False

    def reset(self) -> None:
        self.resampler.reset()
        self.classifier.reset()
        self._recent.clear()
        self.next_seq = 0
        self.finished = False

    def _describe(self, prediction: WindowPrediction) -> dict[str, Any]:
        self._recent.append(prediction.label_index)
        stable = prediction.aggregate_windows == self.spec.aggregation_windows and all(
            label == prediction.aggregate_label_index for label in self._recent
        )
        return {
            **prediction.describe(self.labels, self.spec.sample_rate_hz),
            "aggregate_probabilities": [float(p) for p in prediction.aggregate_probabilities],
            "stable": stable,
            "level_dbfs": _level_dbfs(prediction),
        }

    def _status(self) -> dict[str, Any]:
        return {
            "next_seq": self.next_seq,
            "finished": self.finished,
            "device_samples": self.resampler.samples_received,
            "samples_16k": self.classifier.windower.samples_received,
            "seconds": self.classifier.windower.samples_received / self.spec.sample_rate_hz,
        }

    def push(self, seq: int, body: bytes) -> dict[str, Any]:
        """Accept chunk ``seq``; return the windows it completed."""
        if self.finished:
            raise SessionStateError("the stream was stopped; reset it to start again")
        if seq != self.next_seq:
            raise ChunkOrderError(f"chunk {seq} arrived, but chunk {self.next_seq} is next")
        samples = decode_chunk(body, sample_rate_hz=self.input_rate)
        pcm = self.resampler.push(samples)
        windows = [self._describe(p) for p in self.classifier.push(pcm)]
        self.next_seq += 1
        return {"seq": seq, "windows": windows, **self._status()}

    def stop(self) -> dict[str, Any]:
        """End the stream and return the remaining windows and the final decision."""
        if self.finished:
            raise SessionStateError("the stream was already stopped; reset it to start again")
        predictions = self.classifier.push(self.resampler.flush())
        predictions += self.classifier.flush()
        self.finished = True
        windows = [self._describe(p) for p in predictions]
        decision = self.classifier.decision()
        final = (
            None
            if decision is None
            else {
                "label": decision.label,
                "confidence": decision.confidence,
                "windows": decision.windows,
                "rule": "aggregate of the last emitted window (kw10-streaming-v1)",
            }
        )
        return {"windows": windows, "final_decision": final, **self._status()}
