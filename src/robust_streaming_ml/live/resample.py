"""Deterministic streaming conversion of device audio to 16 kHz 16-bit PCM.

The browser sends mono float32 samples at the device's audio-context rate
``R``. This module converts them to the model's input: mono, 16,000 Hz,
16-bit PCM. It is the only resampling in the live demo, and it is documented
in docs/live-demo.md.

**Resampling** (``R != 16000``). With ``M / L = R / 16000`` in lowest terms, output
sample ``n`` sits at input position ``p = n * M / L``. It is

    y[n] = sum_k x[k] * g_phi(floor(p) - k),   phi = (n * M) mod L

where ``g`` is a Kaiser-windowed sinc lowpass:

    g(t) = c * sinc(c * t) * kaiser(t / T),   c = min(1, 16000 / R)

Here ``T = 16 / c`` input samples (16 zero crossings), the Kaiser window has
``beta = 8``, and each phase's taps are normalised to unit DC gain. Samples
before the start of the stream are zero. On :meth:`StreamingResampler.flush`,
samples after its end are zero, and exactly ``ceil(N * 16000 / R)`` outputs
exist for ``N`` input samples. ``R = 16000`` passes samples through unchanged.

**Quantisation.** Each output is clipped to ``[-1, 32767/32768]``, then rounded
half to even onto the 16-bit grid (``rint(y * 32768)``). The result is
``int16``, exactly like the dataset's PCM.

Outputs are computed in blocks of 160 outputs (10 ms) aligned to absolute
output indices. Each output depends only on its own inputs and a fixed-shape
computation, so the result is bit-identical for any chunking of the input.
This is tested.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
import numpy.typing as npt

__all__ = [
    "BLOCK",
    "KAISER_BETA",
    "OUTPUT_RATE_HZ",
    "SUPPORTED_INPUT_RATES",
    "ZERO_CROSSINGS",
    "ResampleError",
    "StreamingResampler",
    "quantize_pcm16",
    "resample_all",
]

OUTPUT_RATE_HZ = 16000
SUPPORTED_INPUT_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)
ZERO_CROSSINGS = 16
KAISER_BETA = 8.0
BLOCK = 160

Float32 = npt.NDArray[np.float32]
Float64 = npt.NDArray[np.float64]
Int16 = npt.NDArray[np.int16]


class ResampleError(ValueError):
    """Unsupported rate or invalid samples."""


def quantize_pcm16(samples: Float64) -> Int16:
    """Clip to ``[-1, 32767/32768]`` and round half to even onto the int16 grid."""
    clipped = np.clip(samples, -1.0, 32767.0 / 32768.0)
    return np.rint(clipped * 32768.0).astype(np.int16)


def _phase_taps(ratio: Fraction, output_rate: int, input_rate: int) -> tuple[Float64, int]:
    """Per-phase taps ``[L, 2J + 1]`` (index ``d + J`` for offset ``d``) and ``J``."""
    up = ratio.denominator  # ratio = M / L = input_rate / output_rate
    cutoff = min(1.0, output_rate / input_rate)
    half_width = ZERO_CROSSINGS / cutoff
    reach = math.ceil(half_width)
    offsets = np.arange(-reach, reach + 1, dtype=np.float64)
    phases = np.arange(up, dtype=np.float64)[:, None] / up
    t = phases + offsets[None, :]  # position of tap d relative to the output: p - k = phi/L + d
    u = t / half_width
    window = np.where(
        np.abs(u) < 1.0,
        np.i0(KAISER_BETA * np.sqrt(np.clip(1.0 - u * u, 0.0, None))) / np.i0(KAISER_BETA),
        0.0,
    )
    taps = cutoff * np.sinc(cutoff * t) * window
    taps /= taps.sum(axis=1, keepdims=True)
    return np.ascontiguousarray(taps, dtype=np.float64), reach


class StreamingResampler:
    """Chunk-invariant conversion from ``input_rate`` to 16 kHz int16 PCM."""

    _buffer: Float64  # any length; declared so the initial array is not shape-typed

    def __init__(self, input_rate: int) -> None:
        if input_rate not in SUPPORTED_INPUT_RATES:
            raise ResampleError(
                f"unsupported device sample rate {input_rate} Hz; supported: "
                f"{', '.join(str(r) for r in SUPPORTED_INPUT_RATES)}"
            )
        self.input_rate = input_rate
        self.ratio = Fraction(input_rate, OUTPUT_RATE_HZ)
        self.identity = input_rate == OUTPUT_RATE_HZ
        self._taps, self._reach = (
            (np.ones((1, 1)), 0)
            if self.identity
            else _phase_taps(self.ratio, OUTPUT_RATE_HZ, input_rate)
        )
        self.reset()

    def reset(self) -> None:
        self._received = 0
        self._next_output = 0
        # Buffer covers input indices [self._start, self._received); k < 0 is zero.
        self._start = -self._reach
        self._buffer = np.zeros(self._reach, dtype=np.float64)
        self._finished = False

    @property
    def samples_received(self) -> int:
        return self._received

    @property
    def samples_emitted(self) -> int:
        return self._next_output

    def _base(self, n: npt.NDArray[np.int64]) -> npt.NDArray[np.int64]:
        return (n * self.ratio.numerator) // self.ratio.denominator

    def _needed_input(self, last_output: int) -> int:
        """Input samples required (exclusive end) to compute output ``last_output``."""
        base = last_output * self.ratio.numerator // self.ratio.denominator
        return base + self._reach + 1

    def _compute(self, start: int, stop: int) -> Float64:
        n = np.arange(start, stop, dtype=np.int64)
        base = self._base(n)
        phase = (n * self.ratio.numerator) % self.ratio.denominator
        width = 2 * self._reach + 1
        index = base[:, None] + self._reach - np.arange(width, dtype=np.int64)[None, :]
        # taps are stored by offset d = base - k, from -J to J; index column i is d = i - J
        # after reversal: column i holds k = base + J - i, i.e. d = i - J.
        values = self._buffer[index - self._start]
        result: Float64 = (values * self._taps[phase]).sum(axis=1)
        return result

    def _emit(self, final_count: int | None) -> Int16:
        pieces: list[Float64] = []
        while True:
            start = self._next_output
            stop = start - start % BLOCK + BLOCK
            if final_count is not None:
                stop = min(stop, final_count)
                if start >= final_count:
                    break
            elif self._needed_input(stop - 1) > self._received:
                break
            pieces.append(self._compute(start, stop))
            self._next_output = stop
        if not pieces:
            return np.zeros(0, dtype=np.int16)
        # drop input no future output needs
        keep_from = int(self._base(np.array([self._next_output]))[0]) - self._reach
        if keep_from > self._start:
            self._buffer = self._buffer[keep_from - self._start :].copy()
            self._start = keep_from
        return quantize_pcm16(np.concatenate(pieces))

    def push(self, samples: Float32) -> Int16:
        """Append device samples; return the 16 kHz PCM that became computable."""
        if self._finished:
            raise ResampleError("resampler already flushed; reset it to start a new stream")
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ResampleError("samples must be 1-D float32")
        if not np.isfinite(samples).all():
            raise ResampleError("samples contain non-finite values")
        if self.identity:
            self._received += samples.size
            self._next_output += samples.size
            return quantize_pcm16(samples.astype(np.float64))
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float64)])
        self._received += samples.size
        return self._emit(None)

    def flush(self) -> Int16:
        """End the stream: emit the remaining outputs (zeros after the last sample)."""
        if self._finished:
            raise ResampleError("resampler already flushed; reset it to start a new stream")
        self._finished = True
        if self.identity:
            return np.zeros(0, dtype=np.int16)
        total = -(-self._received * self.ratio.denominator // self.ratio.numerator)
        self._buffer = np.concatenate([self._buffer, np.zeros(self._reach + 1, dtype=np.float64)])
        return self._emit(total)


def resample_all(samples: Float32, input_rate: int) -> Int16:
    """The whole signal at once (the offline reference of the streaming conversion)."""
    resampler = StreamingResampler(input_rate)
    return np.concatenate([resampler.push(samples), resampler.flush()])
