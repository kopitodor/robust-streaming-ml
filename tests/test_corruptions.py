from __future__ import annotations

import math
import random
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.typing as npt
import pytest

from robust_streaming_ml.robustness import corruptions as C
from robust_streaming_ml.robustness.corruptions import (
    CORRUPTIONS,
    PROTOCOL_ID,
    PROTOCOL_SEED,
    CodecBackendError,
    CorruptionError,
    codec_backend_info,
    corrupt,
    example_seed,
    get_corruption,
    output_sha256,
    validate_output,
)
from robust_streaming_ml.seed import derive_seed

CONDITIONS = [(spec, sev) for spec in CORRUPTIONS for sev in spec.severities]
IDS = [f"{spec.name}-{sev.name}" for spec, sev in CONDITIONS]
PATH = "yes/aaaaaaaa_nohash_0.wav"


def speech_like(n: int = 12000, seed: int = 0) -> npt.NDArray[np.int16]:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 16000
    env = np.hanning(n)
    x = 8000 * np.sin(2 * np.pi * 300 * t) * env + 3000 * np.sin(2 * np.pi * 3000 * t) * env
    return (x + 300 * rng.standard_normal(n)).astype(np.int16)


# --- protocol definition ----------------------------------------------------------


def test_protocol_defines_exactly_the_five_corruptions_with_pinned_severities() -> None:
    assert PROTOCOL_ID == "kw10-robustness-v1" and PROTOCOL_SEED == 1729
    described = {
        spec.name: [(s.name, dict(s.params)) for s in spec.severities] for spec in CORRUPTIONS
    }
    assert described == {
        "noise": [
            ("snr_20db", {"snr_db": 20.0}),
            ("snr_10db", {"snr_db": 10.0}),
            ("snr_0db", {"snr_db": 0.0}),
        ],
        "reverb": [
            ("rt60_0.3s_drr_+6db", {"rt60_s": 0.3, "drr_db": 6.0}),
            ("rt60_0.6s_drr_0db", {"rt60_s": 0.6, "drr_db": 0.0}),
            ("rt60_1.0s_drr_-6db", {"rt60_s": 1.0, "drr_db": -6.0}),
        ],
        "resample": [
            ("via_8000hz", {"intermediate_rate_hz": 8000.0}),
            ("via_4000hz", {"intermediate_rate_hz": 4000.0}),
            ("via_2000hz", {"intermediate_rate_hz": 2000.0}),
        ],
        "clipping": [
            ("at_0.5_peak", {"threshold_fraction": 0.5}),
            ("at_0.25_peak", {"threshold_fraction": 0.25}),
            ("at_0.1_peak", {"threshold_fraction": 0.1}),
        ],
        "codec": [
            ("opus_level_0.9", {"compression_level": 0.9}),
            ("opus_level_0.95", {"compression_level": 0.95}),
            ("opus_level_1.0", {"compression_level": 1.0}),
        ],
    }
    assert all(spec.version == 1 for spec in CORRUPTIONS)
    assert get_corruption("noise").describe()["severities"][0]["params"] == {"snr_db": 20.0}
    with pytest.raises(CorruptionError):
        get_corruption("wind")
    with pytest.raises(CorruptionError):
        get_corruption("noise").severity("snr_-5db")


def test_example_seed_follows_the_documented_scheme() -> None:
    spec = get_corruption("reverb")
    sev = spec.severities[1]
    expected = derive_seed(1729, "kw10-robustness-v1", "reverb", 1, "rt60_0.6s_drr_0db", PATH)
    assert example_seed(spec, sev, PATH) == expected
    seeds = {
        example_seed(s, v, p) for s, v in CONDITIONS for p in (PATH, "no/bbbbbbbb_nohash_1.wav")
    }
    assert len(seeds) == 2 * len(CONDITIONS)  # distinct per condition and per example


# --- determinism, independence, validity -------------------------------------------


@pytest.mark.parametrize(("spec", "sev"), CONDITIONS, ids=IDS)
def test_every_condition_is_deterministic_valid_and_leaves_input_unchanged(spec, sev) -> None:  # type: ignore[no-untyped-def]
    clip = speech_like()
    before = clip.copy()
    first = corrupt(spec, sev, clip, PATH)
    second = corrupt(spec, sev, clip, PATH)
    assert np.array_equal(clip, before)  # source audio untouched
    assert first.samples.dtype == np.float32 and first.samples.shape == clip.shape
    assert np.isfinite(first.samples).all()
    assert not np.array_equal(first.samples, (clip / 32768.0).astype(np.float32))
    assert np.array_equal(first.samples, second.samples) and first.measures == second.measures
    assert first.seed == example_seed(spec, sev, PATH)
    assert output_sha256(first.samples) == output_sha256(second.samples)
    assert {"input_rms", "output_rms", "output_peak", "over_full_scale_fraction"} <= set(
        first.measures
    )


@pytest.mark.parametrize("name", ["noise", "reverb"])
def test_random_corruptions_depend_on_example_identity(name: str) -> None:
    spec = get_corruption(name)
    clip = speech_like()
    a = corrupt(spec, spec.severities[1], clip, PATH).samples
    b = corrupt(spec, spec.severities[1], clip, "yes/bbbbbbbb_nohash_0.wav").samples
    assert not np.array_equal(a, b)


def test_results_do_not_depend_on_order_or_worker_count() -> None:
    clips = {f"yes/{i:08x}_nohash_0.wav": speech_like(4000 + 97 * i, seed=i) for i in range(8)}
    spec = get_corruption("noise")
    sev = spec.severities[2]
    serial = {p: output_sha256(corrupt(spec, sev, c, p).samples) for p, c in clips.items()}
    paths = list(clips)
    random.Random(0).shuffle(paths)
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes = list(
            pool.map(lambda p: output_sha256(corrupt(spec, sev, clips[p], p).samples), paths)
        )
    assert dict(zip(paths, hashes, strict=True)) == serial


# --- transform-specific behaviour --------------------------------------------------


@pytest.mark.parametrize("sev", get_corruption("noise").severities, ids=lambda s: s.name)
def test_noise_hits_the_exact_snr(sev) -> None:  # type: ignore[no-untyped-def]
    clip = speech_like()
    out = corrupt(get_corruption("noise"), sev, clip, PATH)
    x = clip / 32768.0
    noise = out.samples.astype(np.float64) - x
    snr = 10 * math.log10(np.mean(x * x) / np.mean(noise * noise))
    assert snr == pytest.approx(sev.params["snr_db"], abs=1e-3)  # float32 rounding only
    assert out.measures["achieved_snr_db"] == pytest.approx(sev.params["snr_db"], abs=1e-9)


@pytest.mark.parametrize("sev", get_corruption("reverb").severities, ids=lambda s: s.name)
def test_reverb_preserves_level_and_realises_the_drr(sev) -> None:  # type: ignore[no-untyped-def]
    out = corrupt(get_corruption("reverb"), sev, speech_like(), PATH)
    assert out.measures["output_rms"] == pytest.approx(out.measures["input_rms"], rel=1e-5)
    assert out.measures["realised_drr_db"] == pytest.approx(sev.params["drr_db"], abs=1e-9)
    assert out.measures["rir_length"] == round(sev.params["rt60_s"] * 16000)


@pytest.mark.parametrize("sev", get_corruption("resample").severities, ids=lambda s: s.name)
def test_resampling_removes_content_above_the_intermediate_nyquist(sev) -> None:  # type: ignore[no-untyped-def]
    rate = sev.params["intermediate_rate_hz"]
    n = 16000
    t = np.arange(n) / 16000
    clip = (8000 * np.sin(2 * np.pi * 250 * t) + 8000 * np.sin(2 * np.pi * 0.75 * 8000 * t)).astype(
        np.int16
    )
    out = corrupt(get_corruption("resample"), sev, clip, PATH).samples.astype(np.float64)
    spectrum = np.abs(np.fft.rfft(out[1000:-1000]))
    freqs = np.fft.rfftfreq(n - 2000, 1 / 16000)
    above = spectrum[freqs > rate / 2 * 1.2].max()
    low = spectrum[np.argmin(np.abs(freqs - 250))]
    assert above < 1e-3 * low  # the 6 kHz tone is gone (stopband), 250 Hz kept
    mid = out[2000:-2000]
    reference = (clip[2000:-2000] / 32768.0) if rate >= 12000 else None
    assert reference is None or np.allclose(mid, reference, atol=0.05)


def test_resampling_keeps_dc_gain() -> None:
    clip = np.full(8000, 10000, dtype=np.int16)
    out = corrupt(get_corruption("resample"), get_corruption("resample").severities[0], clip, PATH)
    assert np.allclose(out.samples[1000:-1000], 10000 / 32768, rtol=1e-4)


def test_clipping_by_hand() -> None:
    clip = np.array([3277, 16384, -32768, 0], dtype=np.int16)  # x = 0.1000061, 0.5, -1.0, 0
    out = corrupt(get_corruption("clipping"), get_corruption("clipping").severities[0], clip, PATH)
    # f = 0.5, peak = 1.0, c = 0.5 -> clip to [0.1000061, 0.5, -0.5, 0] -> / 0.5
    np.testing.assert_allclose(out.samples, [3277 / 16384, 1.0, -1.0, 0.0], rtol=1e-7)
    assert out.measures["clipped_fraction"] == 0.25  # only -1.0 exceeds 0.5 in magnitude
    assert out.measures["output_peak"] == 1.0


def test_codec_reports_backend_and_bitrate_falls_with_severity() -> None:
    info = codec_backend_info()
    assert info == {
        "backend": "soundfile/libsndfile",
        "soundfile": "0.14.0",
        "libsndfile": "1.2.2",
        "container": "OGG",
        "codec": "OPUS",
    }
    spec = get_corruption("codec")
    kbps = [
        corrupt(spec, sev, speech_like(16000), PATH).measures["encoded_kbit_per_s"]
        for sev in spec.severities
    ]
    assert kbps[0] > kbps[1] > kbps[2] > 0


# --- failure behaviour ----------------------------------------------------------------


@pytest.mark.parametrize("name", ["noise", "reverb", "resample", "clipping"])
def test_all_zero_clip_fails_instead_of_passing_through(name: str) -> None:
    spec = get_corruption(name)
    with pytest.raises(CorruptionError):
        corrupt(spec, spec.severities[0], np.zeros(4000, dtype=np.int16), PATH)


def test_codec_on_silence_emits_decoder_floor_not_an_error() -> None:
    # Opus decodes digital silence to a tiny non-zero floor (at most 1 LSB of int16),
    # which is a real codec effect rather than a pass-through.
    spec = get_corruption("codec")
    for sev in spec.severities:
        out = corrupt(spec, sev, np.zeros(4000, dtype=np.int16), PATH)
        assert 0.0 < float(np.max(np.abs(out.samples))) <= 1 / 32768


def test_invalid_inputs_are_rejected() -> None:
    spec = get_corruption("noise")
    with pytest.raises(CorruptionError):
        corrupt(spec, spec.severities[0], speech_like().astype(np.int32), PATH)
    with pytest.raises(CorruptionError):
        corrupt(spec, spec.severities[0], np.zeros(0, dtype=np.int16), PATH)


def test_validate_output_rejects_bad_outputs() -> None:
    clean = speech_like(100)
    with pytest.raises(CorruptionError, match="shape"):
        validate_output(np.zeros(99), clean)
    with pytest.raises(CorruptionError, match="non-finite"):
        validate_output(np.full(100, np.nan), clean)
    with pytest.raises(CorruptionError, match="no effect"):
        validate_output(clean / 32768.0, clean)


def test_codec_backend_problems_fail_with_a_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    import soundfile

    monkeypatch.setattr(soundfile, "__libsndfile_version__", "1.0.31")
    with pytest.raises(CodecBackendError, match="pins"):
        codec_backend_info()
    monkeypatch.undo()
    monkeypatch.setattr(soundfile, "available_subtypes", lambda fmt: {"VORBIS": "Vorbis"})
    with pytest.raises(CodecBackendError, match="without Ogg Opus"):
        corrupt(get_corruption("codec"), get_corruption("codec").severities[0], speech_like(), PATH)
    monkeypatch.undo()
    monkeypatch.setitem(sys.modules, "soundfile", None)
    with pytest.raises(CodecBackendError, match="uv sync"):
        C.require_codec_backend()
