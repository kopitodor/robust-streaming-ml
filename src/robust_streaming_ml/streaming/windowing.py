"""Streaming windows for protocol ``kw10-streaming-v1`` (docs/streaming-onnx-protocol.md, §2).

A stream is mono audio at the checkpoint window's sample rate, delivered in
chunks of any length. Chunks are ``int16`` PCM (converted exactly as
``float32(x) / 32768``, like the validated front end) or ``float32`` samples
already on that scale; anything else is rejected.

Window ``k`` covers stream samples ``[k*H, k*H + W)``. There is no startup
padding: window ``k`` is emitted as soon as ``k*H + W`` samples have arrived.
:meth:`StreamingWindower.flush` ends the stream and emits exactly one
zero-padded final window if some received sample is not in any emitted
window. A stream shorter than ``W`` therefore yields one window (the audio
followed by zeros: the clip-level window of ``kw10-clean-eval-v1``), and an
empty stream yields none.

:func:`offline_windows` computes the same windows from a complete signal by
an independent implementation; streaming output must be bit-identical to it
for any chunking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.training.config import WindowSection

__all__ = [
    "AGGREGATION_WINDOWS",
    "HOP_SAMPLES",
    "STREAMING_PROTOCOL_ID",
    "StreamingError",
    "StreamingSpec",
    "StreamingWindower",
    "Window",
    "offline_windows",
    "stream_samples",
]

STREAMING_PROTOCOL_ID = "kw10-streaming-v1"
HOP_SAMPLES = 1600
AGGREGATION_WINDOWS = 5
_PCM16_SCALE = np.float32(32768.0)

Float32 = npt.NDArray[np.float32]


class StreamingError(ValueError):
    """Input or call order violates the streaming contract."""


@dataclass(frozen=True)
class StreamingSpec:
    """Fixed streaming parameters; ``from_window`` binds them to a checkpoint's window."""

    sample_rate_hz: int
    window_samples: int
    hop_samples: int = HOP_SAMPLES
    aggregation_windows: int = AGGREGATION_WINDOWS
    protocol: str = STREAMING_PROTOCOL_ID

    def __post_init__(self) -> None:
        if min(self.sample_rate_hz, self.window_samples, self.hop_samples) < 1:
            raise StreamingError("sample rate, window, and hop must be positive")
        if self.hop_samples > self.window_samples:
            raise StreamingError("hop must not exceed the window (every sample must be covered)")
        if self.aggregation_windows < 1:
            raise StreamingError("aggregation needs at least one window")

    @classmethod
    def from_window(cls, window: WindowSection) -> StreamingSpec:
        if window.alignment != "start_zero_pad":
            raise StreamingError(f"unsupported window alignment {window.alignment!r}")
        return cls(sample_rate_hz=window.sample_rate_hz, window_samples=window.num_samples)

    def describe(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "sample_rate_hz": self.sample_rate_hz,
            "window_samples": self.window_samples,
            "hop_samples": self.hop_samples,
            "aggregation_windows": self.aggregation_windows,
            "startup_padding": "none",
            "final_window": "one zero-padded window if received samples remain uncovered",
            "inference_batch_size": 1,
            "probabilities": "raw softmax (T = 1), float64",
            "aggregation": "mean of the probability vectors of the last aggregation_windows "
            "windows, causal",
        }


@dataclass(frozen=True)
class Window:
    """One analysis window and its position in the stream (sample indices).

    ``emitted_at_sample`` is the stream position at which the window became
    available: ``start_sample + window_samples`` for a full window (it is
    emitted by the push that delivers that sample, however large the chunk),
    and the stream length for the final partial window. Like every other
    field, it does not depend on how the audio was chunked.
    """

    index: int
    start_sample: int
    valid_samples: int
    final_partial: bool
    emitted_at_sample: int
    samples: Float32

    @property
    def end_sample(self) -> int:
        """Exclusive end of the window's span (including any padding)."""
        return self.start_sample + int(self.samples.shape[0])

    def timing(self, sample_rate_hz: int) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "valid_samples": self.valid_samples,
            "final_partial": self.final_partial,
            "emitted_at_sample": self.emitted_at_sample,
            "start_s": self.start_sample / sample_rate_hz,
            "end_s": self.end_sample / sample_rate_hz,
        }


def stream_samples(chunk: Any) -> Float32:
    """Validate one chunk and return it as float32 on the ``int16 / 32768`` scale."""
    if not isinstance(chunk, np.ndarray):
        raise StreamingError(f"a chunk must be a NumPy array, got {type(chunk).__name__}")
    if chunk.ndim != 1:
        raise StreamingError(f"a chunk must be 1-D mono samples, got shape {chunk.shape}")
    if chunk.dtype == np.int16:
        return chunk.astype(np.float32) / _PCM16_SCALE
    if chunk.dtype == np.float32:
        if not np.isfinite(chunk).all():
            raise StreamingError("float32 chunk contains non-finite samples")
        return chunk.copy()
    raise StreamingError(f"chunk dtype must be int16 or float32, got {chunk.dtype}")


def _frozen(samples: Float32) -> Float32:
    samples.setflags(write=False)
    return samples


class StreamingWindower:
    """Incremental windowing of one stream (see the module docstring)."""

    def __init__(self, spec: StreamingSpec, *, sample_rate_hz: int) -> None:
        if sample_rate_hz != spec.sample_rate_hz:
            raise StreamingError(
                f"stream is {sample_rate_hz} Hz but the model expects {spec.sample_rate_hz} Hz; "
                "resampling is not part of the protocol"
            )
        self.spec = spec
        self.reset()

    def reset(self) -> None:
        """Forget all audio and windows; the next sample is index 0 again."""
        self._buffer = np.zeros(0, dtype=np.float32)
        self._buffer_start = 0  # stream index of self._buffer[0]
        self._received = 0
        self._next_index = 0
        self._finished = False

    @property
    def samples_received(self) -> int:
        return self._received

    @property
    def windows_emitted(self) -> int:
        return self._next_index

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def buffered_samples(self) -> int:
        return int(self._buffer.shape[0])

    def _start(self, index: int) -> int:
        return index * self.spec.hop_samples

    def _slice(self, start: int, stop: int) -> Float32:
        return self._buffer[start - self._buffer_start : stop - self._buffer_start]

    def push(self, chunk: Any) -> list[Window]:
        """Append samples; return the windows that became complete, in order."""
        if self._finished:
            raise StreamingError("stream already flushed; call reset() to start a new stream")
        samples = stream_samples(chunk)
        if samples.size == 0:
            return []
        self._buffer = np.concatenate([self._buffer, samples])
        self._received += int(samples.size)
        size = self.spec.window_samples
        emitted: list[Window] = []
        while self._start(self._next_index) + size <= self._received:
            start = self._start(self._next_index)
            emitted.append(
                Window(
                    index=self._next_index,
                    start_sample=start,
                    valid_samples=size,
                    final_partial=False,
                    emitted_at_sample=start + size,
                    samples=_frozen(self._slice(start, start + size).copy()),
                )
            )
            self._next_index += 1
        # Keep only samples that a future window can still use.
        keep_from = self._start(self._next_index)
        if keep_from > self._buffer_start:
            self._buffer = self._buffer[keep_from - self._buffer_start :].copy()
            self._buffer_start = keep_from
        return emitted

    def flush(self) -> list[Window]:
        """End the stream: emit the zero-padded final window if audio is uncovered."""
        if self._finished:
            raise StreamingError("stream already flushed; call reset() to start a new stream")
        self._finished = True
        size = self.spec.window_samples
        index = self._next_index
        covered = self._start(index - 1) + size if index else 0
        if self._received <= covered:
            return []
        start = self._start(index)
        valid = self._received - start
        samples = np.zeros(size, dtype=np.float32)
        samples[:valid] = self._slice(start, self._received)
        self._next_index += 1
        return [
            Window(
                index=index,
                start_sample=start,
                valid_samples=valid,
                final_partial=True,
                emitted_at_sample=self._received,
                samples=_frozen(samples),
            )
        ]


def offline_windows(samples: Any, spec: StreamingSpec) -> list[Window]:
    """Reference windows of a complete signal (independent of :class:`StreamingWindower`)."""
    signal = stream_samples(samples)
    n = int(signal.size)
    size, hop = spec.window_samples, spec.hop_samples
    full = (n - size) // hop + 1 if n >= size else 0
    windows = [
        Window(
            index=k,
            start_sample=k * hop,
            valid_samples=size,
            final_partial=False,
            emitted_at_sample=k * hop + size,
            samples=_frozen(signal[k * hop : k * hop + size].copy()),
        )
        for k in range(full)
    ]
    last_end = (full - 1) * hop + size if full else 0
    if n > last_end:
        start = full * hop
        padded = np.zeros(size, dtype=np.float32)
        padded[: n - start] = signal[start:]
        windows.append(
            Window(
                index=full,
                start_sample=start,
                valid_samples=n - start,
                final_partial=True,
                emitted_at_sample=n,
                samples=_frozen(padded),
            )
        )
    return windows
