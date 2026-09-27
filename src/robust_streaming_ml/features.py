"""Fixed audio window and log-mel feature extraction (feature version 1).

Pipeline for one clip (all steps deterministic on CPU):

1. **Window** (:func:`apply_window`): int16 PCM samples are placed at the start
   of a zero buffer of ``window.num_samples``. A clip longer than the window
   raises :class:`WindowError`; nothing is cropped or resampled.
2. **Scaling**: ``float32(sample) / 32768`` -> values in ``[-1, 1)``.
3. **STFT**: ``torch.stft`` with ``n_fft``, ``hop_length``, ``win_length``, a
   periodic Hann window, ``center=True`` and zero (``constant``) padding of
   ``n_fft // 2`` on both sides. Frames = ``1 + num_samples // hop_length``.
4. **Power**: ``|X|^2`` for the ``n_fft // 2 + 1`` one-sided bins.
5. **Mel**: a triangular filterbank on the HTK mel scale
   (``mel = 2595 * log10(1 + f / 700)``), ``n_mels + 2`` edge points equally
   spaced in mel between ``f_min_hz`` and ``f_max_hz``, filters evaluated at the
   bin frequencies ``k * sample_rate / n_fft``, peak value 1, no area
   normalisation. Computed in float64, stored as float32. Every filter must
   cover at least one bin.
6. **Log**: ``log(mel_power + log_offset)`` (natural log).

Output: ``[batch, 1, n_mels, frames]`` float32; with the defaults
(16 kHz, 16000 samples, n_fft 512, hop 160, 40 mels) that is ``[B, 1, 40, 101]``.

:func:`feature_fingerprint` hashes the settings together with the exact
window and filterbank tensors, so a checkpoint can detect a changed feature
implementation.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from robust_streaming_ml.training.config import FeatureSection, WindowSection

__all__ = [
    "FEATURE_VERSION",
    "LogMelSpectrogram",
    "WindowError",
    "apply_window",
    "feature_fingerprint",
    "mel_filterbank",
    "num_frames",
    "pcm16_to_float",
]

FEATURE_VERSION = 1
_PCM16_SCALE = 32768.0


class WindowError(ValueError):
    """Raised when a clip does not fit the fixed window."""


def apply_window(samples: npt.NDArray[np.int16], window: WindowSection) -> npt.NDArray[np.int16]:
    """Place ``samples`` at the start of a zero buffer of ``window.num_samples``."""
    if samples.ndim != 1:
        raise WindowError(f"expected mono samples, got shape {samples.shape}")
    if samples.shape[0] > window.num_samples:
        raise WindowError(
            f"clip has {samples.shape[0]} samples, window holds {window.num_samples}; "
            "cropping is not allowed"
        )
    out = np.zeros(window.num_samples, dtype=np.int16)
    out[: samples.shape[0]] = samples
    return out


def pcm16_to_float(samples: torch.Tensor) -> torch.Tensor:
    """int16 PCM -> float32 in [-1, 1)."""
    if samples.dtype != torch.int16:
        raise TypeError(f"expected int16 samples, got {samples.dtype}")
    return samples.to(torch.float32) / _PCM16_SCALE


def num_frames(window: WindowSection, features: FeatureSection) -> int:
    """STFT frame count for the fixed window (``center=True``)."""
    return 1 + window.num_samples // features.hop_length


def _hz_to_mel(hz: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(features: FeatureSection, sample_rate_hz: int) -> torch.Tensor:
    """``[n_mels, n_fft // 2 + 1]`` float32 triangular HTK-mel filterbank."""
    n_bins = features.n_fft // 2 + 1
    bin_hz = np.arange(n_bins, dtype=np.float64) * sample_rate_hz / features.n_fft
    edges_mel = np.linspace(
        _hz_to_mel(np.array(features.f_min_hz, dtype=np.float64)),
        _hz_to_mel(np.array(features.f_max_hz, dtype=np.float64)),
        features.n_mels + 2,
        dtype=np.float64,
    )
    edges_hz = _mel_to_hz(edges_mel)
    lower, center, upper = edges_hz[:-2, None], edges_hz[1:-1, None], edges_hz[2:, None]
    rising = (bin_hz[None, :] - lower) / (center - lower)
    falling = (upper - bin_hz[None, :]) / (upper - center)
    bank = np.maximum(0.0, np.minimum(rising, falling))
    empty = [i for i, row in enumerate(bank) if not row.any()]
    if empty:
        raise ValueError(f"mel filters {empty} cover no FFT bin; increase n_fft or reduce n_mels")
    return torch.from_numpy(bank.astype(np.float32))


class LogMelSpectrogram(nn.Module):
    """Waveform ``[B, num_samples]`` float32 -> log-mel ``[B, 1, n_mels, frames]``.

    Has no trainable parameters; the Hann window and mel filterbank are buffers.
    """

    window_fn: torch.Tensor
    filterbank: torch.Tensor

    def __init__(self, features: FeatureSection, window: WindowSection) -> None:
        super().__init__()
        self.features = features
        self.window = window
        self.register_buffer(
            "window_fn", torch.hann_window(features.win_length, periodic=True, dtype=torch.float32)
        )
        self.register_buffer("filterbank", mel_filterbank(features, window.sample_rate_hz))

    @property
    def output_shape(self) -> tuple[int, int, int]:
        return (1, self.features.n_mels, num_frames(self.window, self.features))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 2 or waveform.shape[1] != self.window.num_samples:
            raise ValueError(
                f"expected [batch, {self.window.num_samples}] waveforms, got {list(waveform.shape)}"
            )
        spectrum = torch.stft(
            waveform,
            n_fft=self.features.n_fft,
            hop_length=self.features.hop_length,
            win_length=self.features.win_length,
            window=self.window_fn,
            center=True,
            pad_mode="constant",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        power = spectrum.real.square() + spectrum.imag.square()  # [B, bins, frames]
        mel = torch.matmul(self.filterbank, power)  # [B, n_mels, frames]
        return torch.log(mel + self.features.log_offset).unsqueeze(1)


def feature_fingerprint(features: FeatureSection, window: WindowSection) -> str:
    """SHA-256 over the feature/window settings and the exact window and filterbank."""
    frontend = LogMelSpectrogram(features, window)
    digest = hashlib.sha256()
    settings = {
        "features": features.model_dump(mode="json"),
        "window": window.model_dump(mode="json"),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode("utf-8"))
    for tensor in (frontend.window_fn, frontend.filterbank):
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()
