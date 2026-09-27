from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from robust_streaming_ml.features import (
    LogMelSpectrogram,
    WindowError,
    apply_window,
    feature_fingerprint,
    mel_filterbank,
    num_frames,
    pcm16_to_float,
)
from robust_streaming_ml.training.config import FeatureSection, WindowSection

WINDOW = WindowSection()
FEATURES = FeatureSection()


def test_default_output_shape_is_1_by_40_by_101() -> None:
    frontend = LogMelSpectrogram(FEATURES, WINDOW)
    out = frontend(torch.zeros(3, 16000))
    assert out.shape == (3, 1, 40, 101)
    assert out.dtype == torch.float32
    assert frontend.output_shape == (1, 40, 101)
    assert num_frames(WINDOW, FEATURES) == 101


@pytest.mark.parametrize(
    ("hop", "n_mels", "frames"), [(160, 40, 101), (320, 32, 51), (128, 64, 126)]
)
def test_shapes_follow_the_settings(hop: int, n_mels: int, frames: int) -> None:
    features = FeatureSection(hop_length=hop, n_mels=n_mels)
    out = LogMelSpectrogram(features, WINDOW)(torch.zeros(2, 16000))
    assert out.shape == (2, 1, n_mels, frames)


def test_frontend_has_no_trainable_parameters() -> None:
    frontend = LogMelSpectrogram(FEATURES, WINDOW)
    assert list(frontend.parameters()) == []
    assert set(dict(frontend.named_buffers())) == {"window_fn", "filterbank"}


def test_features_are_deterministic_and_batch_independent() -> None:
    frontend = LogMelSpectrogram(FEATURES, WINDOW)
    x = torch.randn(4, 16000, generator=torch.Generator().manual_seed(0)) * 0.1
    first, second = frontend(x), frontend(x.clone())
    assert torch.equal(first, second)
    single = torch.cat([frontend(x[i : i + 1]) for i in range(4)])
    torch.testing.assert_close(single, first, rtol=1e-5, atol=1e-5)


def test_silence_maps_to_log_offset() -> None:
    out = LogMelSpectrogram(FEATURES, WINDOW)(torch.zeros(1, 16000))
    torch.testing.assert_close(out, torch.full_like(out, math.log(FEATURES.log_offset)))


def test_pure_tone_peaks_in_the_matching_mel_band() -> None:
    t = torch.arange(16000, dtype=torch.float32) / 16000
    tone = 0.5 * torch.sin(2 * math.pi * 1000.0 * t)
    frontend = LogMelSpectrogram(FEATURES, WINDOW)
    band = int(frontend(tone[None]).mean(dim=-1).squeeze().argmax())
    bank = frontend.filterbank
    bin_1khz = round(1000.0 * FEATURES.n_fft / 16000)
    assert band == int(bank[:, bin_1khz].argmax())


def test_filterbank_properties() -> None:
    bank = mel_filterbank(FEATURES, 16000)
    assert bank.shape == (40, 257) and bank.dtype == torch.float32
    assert float(bank.min()) >= 0.0 and float(bank.max()) <= 1.0
    assert bool((bank.sum(dim=1) > 0).all())
    peaks = bank.argmax(dim=1)
    assert bool((peaks[1:] >= peaks[:-1]).all())  # centres increase with index
    low_passed = mel_filterbank(FeatureSection(f_max_hz=4000.0), 16000)
    assert float(low_passed[:, 129:].sum()) == 0.0  # nothing above f_max (bin 128 = 4 kHz)
    below = int(FEATURES.f_min_hz * FEATURES.n_fft / 16000)
    assert float(bank[:, :below].sum()) == 0.0


def test_filterbank_matches_hand_computed_first_filter() -> None:
    """Filter 0 derived by hand from the HTK formula (guards against formula changes).

    mel(20) = 31.75 and mel(8000) = 2840.02, so the spacing is (2840.02 - 31.75) / 41 = 68.494.
    Filter 0 therefore spans 20 Hz, 65.11 Hz (mel 100.25), and 113.09 Hz (mel 168.74).
    Bins are 31.25 Hz apart: bin 1 = (31.25 - 20) / 45.11 = 0.2494;
    bin 2 = (62.5 - 20) / 45.11 = 0.9421; bin 3 = (113.09 - 93.75) / 47.98 = 0.4031.
    """
    bank = mel_filterbank(FEATURES, 16000).numpy()
    np.testing.assert_allclose(bank[0, :5], [0.0, 0.2494, 0.9421, 0.4028, 0.0], atol=1e-3)
    assert int(np.argmax(bank[0])) == 2


def test_empty_mel_filters_are_rejected() -> None:
    features = FeatureSection(n_fft=64, win_length=64, hop_length=32, n_mels=32, f_min_hz=0.0)
    with pytest.raises(ValueError, match="cover no FFT bin"):
        mel_filterbank(features, 16000)


def test_frontend_rejects_wrong_input_length() -> None:
    frontend = LogMelSpectrogram(FEATURES, WINDOW)
    with pytest.raises(ValueError, match="expected"):
        frontend(torch.zeros(2, 15999))
    with pytest.raises(ValueError, match="expected"):
        frontend(torch.zeros(16000))


def test_apply_window_pads_at_the_end_without_changing_samples() -> None:
    samples = np.arange(1, 101, dtype=np.int16)
    out = apply_window(samples, WINDOW)
    assert out.dtype == np.int16 and out.shape == (16000,)
    np.testing.assert_array_equal(out[:100], samples)
    assert not out[100:].any()
    np.testing.assert_array_equal(apply_window(np.ones(16000, np.int16), WINDOW), 1)


def test_apply_window_never_crops() -> None:
    with pytest.raises(WindowError, match="cropping is not allowed"):
        apply_window(np.zeros(16001, np.int16), WINDOW)
    with pytest.raises(WindowError, match="mono"):
        apply_window(np.zeros((2, 100), np.int16), WINDOW)


def test_pcm16_scaling() -> None:
    x = torch.tensor([-32768, 0, 16384, 32767], dtype=torch.int16)
    out = pcm16_to_float(x)
    assert out.dtype == torch.float32
    assert out.tolist() == [-1.0, 0.0, 0.5, 32767 / 32768]
    with pytest.raises(TypeError):
        pcm16_to_float(x.to(torch.int32))


def test_fingerprint_is_stable_and_sensitive() -> None:
    fp = feature_fingerprint(FEATURES, WINDOW)
    assert fp == feature_fingerprint(FeatureSection(), WindowSection())
    assert fp != feature_fingerprint(FeatureSection(n_mels=32), WINDOW)
    assert fp != feature_fingerprint(FEATURES, WindowSection(num_samples=8000))
    assert len(fp) == 64


@pytest.mark.parametrize(
    "overrides",
    [
        {"win_length": 600},  # > n_fft
        {"hop_length": 500},  # > win_length
        {"f_min_hz": 9000.0, "f_max_hz": 8000.0},
        {"n_mels": 300},  # > 257 bins
        {"n_fft": 8},
        {"log_offset": 0.0},
        {"version": 2},
        {"kind": "mfcc"},
    ],
)
def test_invalid_feature_settings_are_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        FeatureSection.model_validate({**FeatureSection().model_dump(), **overrides})


def test_window_settings_are_validated() -> None:
    with pytest.raises(ValidationError):
        WindowSection.model_validate({"alignment": "center"})
    with pytest.raises(ValidationError):
        WindowSection.model_validate({"num_samples": 0})
