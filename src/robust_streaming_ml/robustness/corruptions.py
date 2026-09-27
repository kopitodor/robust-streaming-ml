"""Deterministic audio corruptions for protocol ``kw10-robustness-v1`` (version 1 each).

Each corruption maps one clean clip (int16 PCM, its own ``n`` frames, 16 kHz,
mono) to a float32 clip of exactly ``n`` frames at 16 kHz. The fixed
zero-pad window is applied afterwards, exactly as on the clean path. All
arithmetic is float64 until a final cast to float32. The source audio is
never modified.

Randomness comes only from ``numpy.random.default_rng(seed)`` (PCG64). The
seed is ``derive_seed(PROTOCOL_SEED, PROTOCOL_ID, corruption, version,
severity, relative_path)``, so each clip's corruption depends only on its
identity and the protocol, not on processing order or worker count.

Transforms (``x`` is the clip as float64 ``int16 / 32768``):

- ``noise`` v1: add white Gaussian noise ``w`` (standard normal from the
  clip's RNG, rescaled to exactly unit mean power) at an exact SNR relative
  to the clip's mean power: ``y = x + sqrt(P_x / 10^(snr/10)) * w``.
- ``reverb`` v1: convolve with a synthetic room impulse response. It has a
  unit direct path at lag 0 and a Gaussian tail over lags
  ``1 .. round(rt60 * fs) - 1`` whose amplitude decays by 60 dB over
  ``rt60``, rescaled so that tail energy = ``10^(-drr/10)`` (the
  direct-to-reverberant ratio). The convolution uses a zero-padded FFT; the
  output is truncated to ``n`` frames and rescaled to the input RMS.
- ``resample`` v1: resample to the intermediate rate and back. The filter is
  a lowpass FIR with ``32*M + 1`` taps,
  ``h = sinc(m/M)/M * kaiser(beta=8)``, normalised to unit DC gain, where
  ``M = 16000 / intermediate_rate``. Downsampling filters (centred, zero
  edges) and keeps every M-th sample; upsampling inserts zeros, multiplies
  by M, filters, and truncates to ``n``.
- ``clipping`` v1: hard-saturate at ``c = f * peak(x)`` and divide by ``f``,
  i.e. apply gain ``1/f`` and saturate at the clip's original peak level.
- ``codec`` v1: Ogg Opus via ``soundfile`` / libsndfile at a fixed
  ``compression_level``. The int16 clip is encoded to an in-memory Ogg Opus
  stream at 16 kHz mono and decoded to float32. The decoded rate, channels,
  and frame count must match exactly. The backend versions are pinned:
  another version is an error, never a silent substitution.

Every output is validated (:func:`validate_output`): float32, exactly ``n``
frames, all finite, and different from the input. A transform that changed
nothing is an error, not a silent weakening.
"""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from robust_streaming_ml.seed import derive_seed

__all__ = [
    "CORRUPTIONS",
    "LIBSNDFILE_VERSION",
    "PROTOCOL_ID",
    "PROTOCOL_SEED",
    "SAMPLE_RATE_HZ",
    "SOUNDFILE_VERSION",
    "CodecBackendError",
    "CorruptedClip",
    "CorruptionError",
    "CorruptionSpec",
    "Severity",
    "codec_backend_info",
    "corrupt",
    "example_seed",
    "get_corruption",
    "output_sha256",
    "require_codec_backend",
    "validate_output",
]

PROTOCOL_ID = "kw10-robustness-v1"
PROTOCOL_SEED = 1729
SAMPLE_RATE_HZ = 16000
SOUNDFILE_VERSION = "0.14.0"
LIBSNDFILE_VERSION = "1.2.2"

Float64 = npt.NDArray[np.float64]
Float32 = npt.NDArray[np.float32]
Int16 = npt.NDArray[np.int16]
Transform = Callable[
    [Int16, Float64, Mapping[str, float], np.random.Generator],
    tuple[npt.NDArray[Any], dict[str, float]],
]


class CorruptionError(RuntimeError):
    """A corruption cannot be applied or produced an invalid output."""


class CodecBackendError(CorruptionError):
    """The pinned codec backend is missing or differs from the protocol."""


@dataclass(frozen=True)
class Severity:
    name: str
    params: Mapping[str, float]


@dataclass(frozen=True)
class CorruptionSpec:
    name: str
    version: int
    description: str
    severities: tuple[Severity, ...]
    transform: Transform

    def severity(self, name: str) -> Severity:
        for severity in self.severities:
            if severity.name == name:
                return severity
        raise CorruptionError(f"{self.name} has no severity {name!r}")

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "severities": [{"name": s.name, "params": dict(s.params)} for s in self.severities],
        }


@dataclass(frozen=True)
class CorruptedClip:
    samples: Float32
    seed: int
    measures: dict[str, float]


def _rms(x: npt.NDArray[Any]) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64))))


# --- transforms -------------------------------------------------------------


def _noise(
    clip: Int16, x: Float64, p: Mapping[str, float], rng: np.random.Generator
) -> tuple[Float64, dict[str, float]]:
    power = float(np.mean(x * x))
    if power == 0.0:
        raise CorruptionError("noise: SNR is undefined for an all-zero clip")
    w = rng.standard_normal(x.size)
    w /= math.sqrt(float(np.mean(w * w)))
    noise = math.sqrt(power / 10.0 ** (p["snr_db"] / 10.0)) * w
    achieved = 10.0 * math.log10(power / float(np.mean(noise * noise)))
    return x + noise, {"achieved_snr_db": achieved}


def _reverb(
    clip: Int16, x: Float64, p: Mapping[str, float], rng: np.random.Generator
) -> tuple[Float64, dict[str, float]]:
    rt60, drr = p["rt60_s"], p["drr_db"]
    length = round(rt60 * SAMPLE_RATE_HZ)
    if length < 2:
        raise CorruptionError("reverb: rt60 too short for a tail")
    lags = np.arange(1, length, dtype=np.float64)
    tail = rng.standard_normal(length - 1) * 10.0 ** (-3.0 * lags / (rt60 * SAMPLE_RATE_HZ))
    tail *= math.sqrt(10.0 ** (-drr / 10.0) / float(np.sum(tail * tail)))
    rir = np.concatenate([[1.0], tail])
    nfft = 1 << (x.size + length - 2).bit_length()
    y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(rir, nfft), nfft)[: x.size]
    in_rms, out_rms = _rms(x), _rms(y)
    if in_rms == 0.0 or out_rms == 0.0:
        raise CorruptionError("reverb: cannot level-match an all-zero clip")
    realised_drr = 10.0 * math.log10(1.0 / float(np.sum(tail * tail)))
    return y * (in_rms / out_rms), {"rir_length": float(length), "realised_drr_db": realised_drr}


def _lowpass(factor: int) -> Float64:
    taps = 32 * factor + 1
    m = np.arange(taps, dtype=np.float64) - (taps - 1) / 2
    h = np.sinc(m / factor) / factor * np.kaiser(taps, 8.0)
    return h / h.sum()


def _resample(
    clip: Int16, x: Float64, p: Mapping[str, float], rng: np.random.Generator
) -> tuple[Float64, dict[str, float]]:
    rate = int(p["intermediate_rate_hz"])
    if SAMPLE_RATE_HZ % rate:
        raise CorruptionError(f"resample: {rate} Hz does not divide {SAMPLE_RATE_HZ} Hz")
    factor = SAMPLE_RATE_HZ // rate
    h = _lowpass(factor)
    centre = (h.size - 1) // 2
    down = np.convolve(x, h)[centre : centre + x.size][::factor]
    up = np.zeros(down.size * factor, dtype=np.float64)
    up[::factor] = down * factor
    y = np.convolve(up, h)[centre : centre + up.size][: x.size]
    return y, {"intermediate_frames": float(down.size), "filter_taps": float(h.size)}


def _clipping(
    clip: Int16, x: Float64, p: Mapping[str, float], rng: np.random.Generator
) -> tuple[Float64, dict[str, float]]:
    fraction = p["threshold_fraction"]
    peak = float(np.max(np.abs(x)))
    if peak == 0.0:
        raise CorruptionError("clipping: an all-zero clip cannot be clipped")
    level = fraction * peak
    clipped = float(np.mean(np.abs(x) > level))
    return np.clip(x, -level, level) / fraction, {"clipped_fraction": clipped}


def codec_backend_info() -> dict[str, str]:
    """Versions of the codec backend, or an error describing why it is unusable."""
    module = require_codec_backend()
    return {
        "backend": "soundfile/libsndfile",
        "soundfile": str(module.__version__),
        "libsndfile": str(module.__libsndfile_version__),
        "container": "OGG",
        "codec": "OPUS",
    }


def require_codec_backend() -> ModuleType:
    """Return the ``soundfile`` module if it is exactly the pinned backend with Opus."""
    try:
        import soundfile
    except (ImportError, OSError) as exc:  # OSError: missing native libsndfile
        raise CodecBackendError(
            "codec corruption needs the 'soundfile' package with its bundled libsndfile "
            f"(pinned soundfile {SOUNDFILE_VERSION}, libsndfile {LIBSNDFILE_VERSION}); "
            f"run 'uv sync'. Import failed: {exc}"
        ) from exc
    found = (str(soundfile.__version__), str(soundfile.__libsndfile_version__))
    if found != (SOUNDFILE_VERSION, LIBSNDFILE_VERSION):
        raise CodecBackendError(
            f"codec backend is soundfile {found[0]} / libsndfile {found[1]}, but the protocol "
            f"pins {SOUNDFILE_VERSION} / {LIBSNDFILE_VERSION}; results would not be comparable"
        )
    if "OPUS" not in soundfile.available_subtypes("OGG"):
        raise CodecBackendError("libsndfile was built without Ogg Opus support")
    module: ModuleType = soundfile
    return module


def _codec(
    clip: Int16, x: Float64, p: Mapping[str, float], rng: np.random.Generator
) -> tuple[Float32, dict[str, float]]:
    sf = require_codec_backend()
    buffer = io.BytesIO()
    sf.write(
        buffer,
        clip,
        SAMPLE_RATE_HZ,
        format="OGG",
        subtype="OPUS",
        compression_level=p["compression_level"],
    )
    encoded = buffer.getvalue()
    decoded, rate = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    if rate != SAMPLE_RATE_HZ or decoded.shape[1] != 1:
        raise CorruptionError(f"codec: decoded {rate} Hz x {decoded.shape[1]} channels")
    if decoded.shape[0] != clip.size:
        raise CorruptionError(f"codec: decoded {decoded.shape[0]} frames for {clip.size}")
    kbps = len(encoded) * 8 / (clip.size / SAMPLE_RATE_HZ) / 1000
    return decoded[:, 0], {"encoded_bytes": float(len(encoded)), "encoded_kbit_per_s": kbps}


CORRUPTIONS: tuple[CorruptionSpec, ...] = (
    CorruptionSpec(
        "noise",
        1,
        "additive white Gaussian noise at an exact SNR relative to the clip's mean power",
        (
            Severity("snr_20db", {"snr_db": 20.0}),
            Severity("snr_10db", {"snr_db": 10.0}),
            Severity("snr_0db", {"snr_db": 0.0}),
        ),
        _noise,
    ),
    CorruptionSpec(
        "reverb",
        1,
        "synthetic exponential-decay RIR (unit direct path + Gaussian tail at a set DRR), "
        "output level-matched to the input RMS",
        (
            Severity("rt60_0.3s_drr_+6db", {"rt60_s": 0.3, "drr_db": 6.0}),
            Severity("rt60_0.6s_drr_0db", {"rt60_s": 0.6, "drr_db": 0.0}),
            Severity("rt60_1.0s_drr_-6db", {"rt60_s": 1.0, "drr_db": -6.0}),
        ),
        _reverb,
    ),
    CorruptionSpec(
        "resample",
        1,
        "16 kHz -> intermediate rate -> 16 kHz with a Kaiser-windowed-sinc FIR (32*M+1 taps)",
        (
            Severity("via_8000hz", {"intermediate_rate_hz": 8000.0}),
            Severity("via_4000hz", {"intermediate_rate_hz": 4000.0}),
            Severity("via_2000hz", {"intermediate_rate_hz": 2000.0}),
        ),
        _resample,
    ),
    CorruptionSpec(
        "clipping",
        1,
        "hard saturation at a fraction of the clip's peak, rescaled to the original peak",
        (
            Severity("at_0.5_peak", {"threshold_fraction": 0.5}),
            Severity("at_0.25_peak", {"threshold_fraction": 0.25}),
            Severity("at_0.1_peak", {"threshold_fraction": 0.1}),
        ),
        _clipping,
    ),
    CorruptionSpec(
        "codec",
        1,
        "Ogg Opus encode/decode via soundfile/libsndfile at a fixed compression_level",
        (
            Severity("opus_level_0.9", {"compression_level": 0.9}),
            Severity("opus_level_0.95", {"compression_level": 0.95}),
            Severity("opus_level_1.0", {"compression_level": 1.0}),
        ),
        _codec,
    ),
)


def get_corruption(name: str) -> CorruptionSpec:
    for spec in CORRUPTIONS:
        if spec.name == name:
            return spec
    raise CorruptionError(f"unknown corruption {name!r}")


def example_seed(spec: CorruptionSpec, severity: Severity, relative_path: str) -> int:
    """Per-example seed from stable identity and the fixed protocol seed."""
    return derive_seed(
        PROTOCOL_SEED, PROTOCOL_ID, spec.name, spec.version, severity.name, relative_path
    )


def output_sha256(samples: Float32) -> str:
    """SHA-256 of the corrupted samples as little-endian float32."""
    return hashlib.sha256(samples.astype("<f4").tobytes()).hexdigest()


def validate_output(samples: npt.NDArray[Any], clean: Int16) -> Float32:
    """Check a transform's output against the protocol; return it as float32."""
    if samples.ndim != 1 or samples.shape[0] != clean.size:
        raise CorruptionError(f"output has shape {samples.shape}, expected ({clean.size},)")
    out = samples.astype(np.float32)
    if not np.isfinite(out).all():
        raise CorruptionError("output contains non-finite samples")
    if np.array_equal(out, (clean.astype(np.float64) / 32768.0).astype(np.float32)):
        raise CorruptionError("corruption had no effect on the clip")
    return out


def corrupt(
    spec: CorruptionSpec, severity: Severity, clean: Int16, relative_path: str
) -> CorruptedClip:
    """Apply one corruption/severity to one clip (pure; the input array is not modified)."""
    if clean.dtype != np.int16 or clean.ndim != 1 or clean.size == 0:
        raise CorruptionError("input must be a non-empty 1-D int16 clip")
    seed = example_seed(spec, severity, relative_path)
    rng = np.random.default_rng(seed)
    x = clean.astype(np.float64) / 32768.0
    raw, measures = spec.transform(clean.copy(), x.copy(), severity.params, rng)
    out = validate_output(np.asarray(raw), clean)
    measures = dict(measures)
    measures.update(
        {
            "input_rms": _rms(x),
            "output_rms": _rms(out),
            "output_peak": float(np.max(np.abs(out))),
            "over_full_scale_fraction": float(np.mean(np.abs(out) > 1.0)),
        }
    )
    return CorruptedClip(samples=out, seed=seed, measures=measures)
