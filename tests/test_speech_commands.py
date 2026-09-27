"""Speech Commands spec consistency and real-data integration validation.

The integration test reads the local extraction under data/raw/ when it is
present and is skipped otherwise; it never downloads anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from robust_streaming_ml.data.speech_commands import LABELS, SPEECH_COMMANDS_KW10
from robust_streaming_ml.data.validation import validate_dataset

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "docs" / "datasets" / "speech_commands_v0.02.md"
CONTRACT = REPO / "docs" / "data-contract.md"
DATA_DIR = Path(os.environ.get("RSML_DATA_DIR", REPO / "data"))
EXTRACTION = DATA_DIR / "raw" / SPEECH_COMMANDS_KW10.archive.extract_dirname
ARCHIVE = DATA_DIR / "raw" / SPEECH_COMMANDS_KW10.archive.filename

# Observed from the archive listing on 2026-09-27 (docs/datasets/speech_commands_v0.02.md).
EXPECTED_COUNTS = {
    "train": {"yes": 3228, "no": 3130, "up": 2948, "down": 3134, "left": 3037,
              "right": 3019, "on": 3086, "off": 2970, "stop": 3111, "go": 3106},
    "validation": {"yes": 397, "no": 406, "up": 350, "down": 377, "left": 352,
                   "right": 363, "on": 363, "off": 373, "stop": 350, "go": 372},
    "test": {"yes": 419, "no": 405, "up": 425, "down": 406, "left": 412,
             "right": 396, "on": 396, "off": 402, "stop": 411, "go": 402},
}  # fmt: skip


def test_labels_match_the_contract() -> None:
    assert LABELS == ("yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
    assert '"yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"' in (
        CONTRACT.read_text(encoding="utf-8")
    )


def test_metadata_matches_the_verification_record() -> None:
    record = RECORD.read_text(encoding="utf-8")
    meta = SPEECH_COMMANDS_KW10.metadata
    for key in ("retrieved_at_utc", "license_url", "license_file_path", "attribution_text"):
        assert meta[key] in record, key
    assert "Creative Commons Attribution 4.0 International (CC BY 4.0)" in record
    assert meta["license_name"] == "Creative Commons Attribution 4.0 International (CC BY 4.0)"
    assert "**Project owner, 2026-09-27.**" in record
    assert (meta["license_verified_by"], meta["license_verified_on"]) == (
        "Project owner",
        "2026-09-27",
    )
    pinned = SPEECH_COMMANDS_KW10.expected_manifest_sha256
    assert pinned is not None and pinned in record


@pytest.mark.integration
@pytest.mark.skipif(
    not (EXTRACTION / ".acquisition.json").is_file() or not ARCHIVE.is_file(),
    reason="local Speech Commands extraction/archive not present",
)
def test_local_extraction_satisfies_the_contract() -> None:
    result = validate_dataset(SPEECH_COMMANDS_KW10, EXTRACTION, ARCHIVE)
    report = result.report
    assert result.passed, report["errors"]["by_code"]
    assert report["counts"]["by_split_label"] == EXPECTED_COUNTS
    assert report["counts"]["total"] == len(result.rows) == 38546
    assert report["speakers"]["in_multiple_splits"] == 0
    assert report["archive_crosscheck"]["label_files_compared"] == 38546
    assert set(report["warnings"]["by_code"]) <= {
        "all_zero_audio",
        "near_silent_audio",
        "duplicate_audio_within_split_label",
    }
    assert result.manifest_sha256 == SPEECH_COMMANDS_KW10.expected_manifest_sha256
