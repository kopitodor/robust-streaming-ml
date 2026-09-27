from __future__ import annotations

import io
import wave

import numpy as np
import pytest

from robust_streaming_ml.data.wav import WavError, parse_wav
from tests.audio_fixtures import tone, wav_bytes


def test_parses_canonical_pcm16_and_matches_stdlib_wave() -> None:
    samples = tone(1234, seed=5)
    data = wav_bytes(samples)
    parsed = parse_wav(data)
    assert parsed.format_tag == 1
    assert parsed.sample_format == "PCM_16"
    assert (parsed.num_channels, parsed.sample_rate_hz, parsed.bits_per_sample) == (1, 16000, 16)
    assert parsed.num_frames == 1234
    np.testing.assert_array_equal(parsed.samples_int16(), samples)
    with wave.open(io.BytesIO(data)) as reference:
        assert reference.getnframes() == 1234
        assert reference.getframerate() == 16000
        assert reference.readframes(1234) == parsed.sample_bytes


def test_extreme_sample_values_decode_exactly() -> None:
    samples = np.array([-32768, 32767, 0, -1, 1], dtype=np.int16)
    np.testing.assert_array_equal(parse_wav(wav_bytes(samples)).samples_int16(), samples)


def test_extra_chunks_are_skipped_and_reported() -> None:
    data = wav_bytes(
        tone(10, 1),
        chunks_before=[(b"JUNK", b"abc")],  # odd size: exercises the pad byte
        chunks_after=[(b"LIST", b"INFOxxxx")],
    )
    parsed = parse_wav(data)
    assert parsed.extra_chunks == ("JUNK", "LIST")
    np.testing.assert_array_equal(parsed.samples_int16(), tone(10, 1))


def test_other_formats_parse_but_are_labelled() -> None:
    stereo = parse_wav(wav_bytes(tone(20, 1), channels=2))
    assert stereo.num_channels == 2 and stereo.num_frames == 10
    float32 = parse_wav(wav_bytes(b"\0" * 16, format_tag=3, bits=32))
    assert float32.sample_format == "FORMAT_TAG_3_32BIT"
    with pytest.raises(WavError, match="cannot decode"):
        float32.samples_int16()
    assert parse_wav(wav_bytes(b"\0" * 6, bits=24)).sample_format == "PCM_24"


def test_zero_frames_is_structurally_valid() -> None:
    assert parse_wav(wav_bytes(b"")).num_frames == 0


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"RIFF", "too short"),
        (wav_bytes(tone(10, 1))[:4].replace(b"RIFF", b"RIFX") + wav_bytes(tone(10, 1))[4:], "RIFF"),
        (wav_bytes(tone(10, 1)).replace(b"WAVE", b"AVI ", 1), "RIFF/WAVE"),
        (wav_bytes(tone(10, 1), riff_size_delta=2), "RIFF size"),
        (wav_bytes(tone(10, 1))[:-4], "RIFF size"),
        (wav_bytes(tone(10, 1), trailing=b"xx", riff_size_delta=2), "extends past end|truncated"),
        (wav_bytes(tone(10, 1), trailing=b"xxxxxxxxxx", riff_size_delta=10), "extends past end"),
        (wav_bytes(tone(10, 1), data_size=100), "extends past end"),
        (wav_bytes(tone(10, 1), order="data"), "data chunk before fmt"),
        (wav_bytes(tone(10, 1), order="fmt"), "missing data"),
        (wav_bytes(tone(10, 1), order="data,fmt"), "data chunk before fmt"),
        (wav_bytes(tone(10, 1), order="fmt,fmt,data"), "more than one fmt"),
        (wav_bytes(tone(10, 1), order="fmt,data,data"), "more than one data"),
        (wav_bytes(tone(10, 1), fmt_size=14), "fmt chunk too short"),
        (wav_bytes(tone(10, 1), block_align=4), "block_align"),
        (wav_bytes(tone(10, 1), byte_rate=16000), "byte_rate"),
        (wav_bytes(b"\0\0\0", riff_size_delta=0), "not a multiple"),
        (wav_bytes(tone(10, 1), channels=0), "invalid fmt"),
        (wav_bytes(tone(10, 1), rate=0), "invalid fmt"),
    ],
)
def test_malformed_files_are_rejected(data: bytes, message: str) -> None:
    with pytest.raises(WavError, match=message):
        parse_wav(data)
