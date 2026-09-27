from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from typing import Any

import pytest

from robust_streaming_ml.data.manifest import (
    COLUMNS,
    Issue,
    ManifestError,
    ManifestRow,
    check_rows,
    decode_manifest,
    encode_manifest,
    format_duration,
    parse_filename,
    read_manifest,
)

LABELS = ("yes", "no")


def issues(rows: list[ManifestRow]) -> list[Issue]:
    return check_rows(
        rows, LABELS, sample_rate_hz=16000, num_channels=1, sample_format="PCM_16", max_frames=16000
    )


def row(**overrides: Any) -> ManifestRow:
    base = ManifestRow(
        relative_path="yes/aaaaaaaa_nohash_0.wav",
        split="train",
        label="yes",
        label_index=0,
        speaker_id="aaaaaaaa",
        utterance_index=0,
        sha256="a" * 64,
        audio_sha256="b" * 64,
        size_bytes=44 + 2 * 16000,
        sample_rate_hz=16000,
        num_channels=1,
        sample_format="PCM_16",
        num_frames=16000,
        duration_s="1.000000",
        peak_abs=1000,
        is_all_zero=False,
    )
    return dataclasses.replace(base, **overrides)


def codes(rows: list[ManifestRow]) -> set[str]:
    return {i.code for i in issues(rows)}


def test_columns_follow_the_contract_order() -> None:
    assert COLUMNS == (
        "relative_path", "split", "label", "label_index", "speaker_id", "utterance_index",
        "sha256", "audio_sha256", "size_bytes", "sample_rate_hz", "num_channels",
        "sample_format", "num_frames", "duration_s", "peak_abs", "is_all_zero",
    )  # fmt: skip


def test_encoding_is_sorted_lf_and_lowercase_booleans() -> None:
    later = row(relative_path="yes/bbbbbbbb_nohash_0.wav", speaker_id="bbbbbbbb")
    zero = row(peak_abs=0, is_all_zero=True)
    data = encode_manifest([later, zero])
    text = data.decode("utf-8")
    lines = text.split("\n")
    assert lines[0] == ",".join(COLUMNS)
    assert lines[1].startswith("yes/aaaaaaaa_nohash_0.wav,train,yes,0,aaaaaaaa,0,")
    assert lines[1].endswith(",16000,1.000000,0,true")
    assert lines[2].endswith(",1000,false")
    assert text.endswith("\n") and "\r" not in text
    assert encode_manifest([zero, later]) == data  # input order does not matter


def test_round_trip_and_read_with_checksum(tmp_path: Path) -> None:
    rows = [row(), row(relative_path="no/cccccccc_nohash_3.wav", label="no", label_index=1,
                       speaker_id="cccccccc", utterance_index=3)]  # fmt: skip
    data = encode_manifest(rows)
    assert decode_manifest(data) == sorted(rows, key=lambda r: r.relative_path)
    path = tmp_path / "m.csv"
    path.write_bytes(data)
    assert read_manifest(path, expected_sha256=hashlib.sha256(data).hexdigest()) == decode_manifest(
        data
    )
    with pytest.raises(ManifestError, match="SHA-256"):
        read_manifest(path, expected_sha256="0" * 64)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda t: t.replace("\n", "\r\n"), "line endings"),
        (lambda t: t.replace("relative_path,", "path,", 1), "header"),
        (lambda t: t.replace(",false\n", ",no\n", 1), "is_all_zero"),
        (lambda t: t.replace(",train,yes,0,", ",train,yes,00,", 1), "canonical integer"),
        (lambda t: t.replace(",train,yes,0,", ",train,yes,x,", 1), "canonical integer"),
        (lambda t: t.replace(",false\n", ",false,extra\n", 1), "expected 16 fields"),
        (
            lambda t: "\n".join([t.split("\n")[0], *reversed(t.split("\n")[1:-1])]) + "\n",
            "canonical",
        ),
    ],
)
def test_decode_rejects_non_canonical_input(mutate: Any, message: str) -> None:
    rows = [row(), row(relative_path="yes/bbbbbbbb_nohash_0.wav", speaker_id="bbbbbbbb")]
    text = encode_manifest(rows).decode("utf-8")
    with pytest.raises(ManifestError, match=message):
        decode_manifest(mutate(text).encode("utf-8"))


def test_decode_rejects_non_utf8() -> None:
    with pytest.raises(ManifestError, match="UTF-8"):
        decode_manifest(b"\xff\xfe")


def test_parse_filename() -> None:
    assert parse_filename("0a1b2c3d_nohash_12.wav") == ("0a1b2c3d", 12)
    for bad in (
        "0a1b2c3d_nohash_01.wav",  # leading zero
        "0A1B2C3D_nohash_1.wav",  # uppercase hex
        "0a1b2c3_nohash_1.wav",  # 7 characters
        "0a1b2c3d_hash_1.wav",
        "0a1b2c3d_nohash_1.WAV",
        "0a1b2c3d_nohash_-1.wav",
        "0a1b2c3d_nohash_1.wav.bak",
    ):
        assert parse_filename(bad) is None, bad


def test_format_duration() -> None:
    assert format_duration(16000, 16000) == "1.000000"
    assert format_duration(8000, 16000) == "0.500000"
    assert format_duration(1, 16000) == "0.000063"  # 0.0000625 rounds up in binary
    assert format_duration(15999, 16000) == "0.999938"


def test_valid_rows_have_no_issues() -> None:
    assert issues([row(), row(relative_path="yes/aaaaaaaa_nohash_1.wav", utterance_index=1)]) == []


@pytest.mark.parametrize(
    ("rows", "code"),
    [
        ([row(), row()], "row_duplicate"),
        ([row(), row(sha256="c" * 64)], "row_duplicate_path"),
        (
            [row(), row(relative_path="yes/AAAAAAAA_nohash_0.wav")],
            "row_case_collision",
        ),
        ([row(split="dev")], "row_invalid_split"),
        ([row(label="maybe")], "row_invalid_label"),
        ([row(label_index=1)], "row_label_index_mismatch"),
        ([row(relative_path="yes/x.wav")], "row_bad_path"),
        ([row(relative_path="a/yes/aaaaaaaa_nohash_0.wav")], "row_bad_path"),
        ([row(relative_path="no/aaaaaaaa_nohash_0.wav")], "row_path_label_mismatch"),
        ([row(speaker_id="bbbbbbbb")], "row_identity_mismatch"),
        ([row(utterance_index=4)], "row_identity_mismatch"),
        ([row(sha256="ABC")], "row_bad_hash"),
        ([row(audio_sha256="g" * 64)], "row_bad_hash"),
        ([row(sample_rate_hz=8000)], "row_bad_format"),
        ([row(num_channels=2)], "row_bad_format"),
        ([row(sample_format="PCM_24")], "row_bad_format"),
        ([row(num_frames=0, duration_s="0.000000")], "row_bad_num_frames"),
        ([row(num_frames=16001, duration_s="1.000063")], "row_bad_num_frames"),
        ([row(duration_s="0.999999")], "row_bad_duration"),
        ([row(size_bytes=100)], "row_bad_size"),
        ([row(peak_abs=0, is_all_zero=False)], "row_bad_peak"),
        ([row(peak_abs=5, is_all_zero=True)], "row_bad_peak"),
        ([row(peak_abs=40000)], "row_bad_peak"),
    ],
)
def test_row_checks_detect_each_violation(rows: list[ManifestRow], code: str) -> None:
    assert code in codes(rows)
